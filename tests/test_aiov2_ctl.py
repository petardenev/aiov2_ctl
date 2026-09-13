"""
Tests for the readsb SDR recovery, --sdr-biastee and PyGPSClient setup.

Everything that touches the board (GPIO, sysfs, systemctl, rtl_biast,
chown) is mocked, so these run anywhere:

    python3 -m unittest discover -s tests
"""
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import aiov2_ctl as a  # noqa: E402


def fake_sysfs(devices):
    """
    Patch os.listdir/read_sysfs to present USB devices.
    devices: {name: {"idVendor": ..., "idProduct": ..., "speed": ..., ...}}
    """
    base = "/sys/bus/usb/devices"

    def read(path):
        dev, attr = os.path.split(path)
        return devices.get(os.path.basename(dev), {}).get(attr)

    def listdir(path):
        assert path == base
        return list(devices)

    return (
        mock.patch.object(a.os, "listdir", side_effect=listdir),
        mock.patch.object(a, "read_sysfs", side_effect=read),
    )


SDR = {"idVendor": "0bda", "idProduct": "2838", "speed": "480", "busnum": "1", "devnum": "17"}
HUB = {"idVendor": "05e3", "idProduct": "0610", "speed": "480", "busnum": "1", "devnum": "2"}


class RtlsdrUsbTests(unittest.TestCase):
    def sysfs(self, devices):
        for patcher in fake_sysfs(devices):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_finds_rtl2832_among_other_devices(self):
        self.sysfs({"1-1": HUB, "1-1.3": SDR})
        self.assertEqual(a.rtlsdr_sysfs(), "/sys/bus/usb/devices/1-1.3")

    def test_accepts_both_rtl_product_ids(self):
        self.sysfs({"1-1.3": dict(SDR, idProduct="2832")})
        self.assertIsNotNone(a.rtlsdr_sysfs())

    def test_no_sdr(self):
        self.sysfs({"1-1": HUB})
        self.assertIsNone(a.rtlsdr_sysfs())
        self.assertFalse(a.rtlsdr_high_speed())
        self.assertFalse(a.rtlsdr_in_use())

    def test_high_speed_only_at_480M(self):
        self.sysfs({"1-1.3": SDR})
        self.assertTrue(a.rtlsdr_high_speed())

    def test_wedged_full_speed_is_not_ready(self):
        self.sysfs({"1-1.3": dict(SDR, speed="12")})
        self.assertFalse(a.rtlsdr_high_speed())

    def test_in_use_checks_the_usbfs_node(self):
        self.sysfs({"1-1.3": SDR})
        with mock.patch.object(a.subprocess, "call", return_value=0) as call:
            self.assertTrue(a.rtlsdr_in_use())
        self.assertEqual(call.call_args[0][0], ["fuser", "-s", "/dev/bus/usb/001/017"])

    def test_not_in_use_when_fuser_finds_nothing(self):
        self.sysfs({"1-1.3": SDR})
        with mock.patch.object(a.subprocess, "call", return_value=1):
            self.assertFalse(a.rtlsdr_in_use())

    def test_not_in_use_when_fuser_is_missing(self):
        self.sysfs({"1-1.3": SDR})
        with mock.patch.object(a.subprocess, "call", side_effect=FileNotFoundError):
            self.assertFalse(a.rtlsdr_in_use())

    def test_wait_for_rtlsdr_uses_custom_check(self):
        check = mock.Mock(side_effect=[False, False, True])
        with mock.patch.object(a.time, "sleep"):
            self.assertTrue(a.wait_for_rtlsdr(5, check))
        self.assertEqual(check.call_count, 3)

    def test_wait_for_rtlsdr_times_out(self):
        with mock.patch.object(a.time, "sleep"):
            self.assertFalse(a.wait_for_rtlsdr(3, lambda: False))

    def test_wait_for_rtlsdr_defaults_to_lsusb_presence(self):
        with mock.patch.object(a.time, "sleep"), \
                mock.patch.object(a, "rtlsdr_present", return_value=True) as present:
            self.assertTrue(a.wait_for_rtlsdr(2))
        present.assert_called()


class ReadsbStalledTests(unittest.TestCase):
    NOW = 1_000_000.0

    def stalled(self, stats=None, running=True, pid="1234", uptime="600", stats_error=None):
        def run(cmd):
            return {"systemctl": pid, "ps": uptime}[cmd[0]]

        if stats_error:
            opener = mock.patch("builtins.open", side_effect=stats_error)
        else:
            opener = mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(stats)))

        with mock.patch.object(a, "readsb_service_running", return_value=running), \
                mock.patch.object(a.GpioController, "run", side_effect=run), \
                mock.patch.object(a.time, "time", return_value=self.NOW), \
                opener:
            return a.readsb_stalled()

    def stats(self, age=1, samples=18_000_000):
        return {"now": self.NOW - age, "last1min": {"local": {"samples_processed": samples}}}

    def test_healthy(self):
        self.assertFalse(self.stalled(self.stats()))

    def test_not_running_is_not_stalled(self):
        self.assertFalse(self.stalled(self.stats(age=999), running=False))

    def test_fresh_start_gets_a_grace_period(self):
        self.assertFalse(self.stalled(self.stats(age=999, samples=0), uptime="30"))

    def test_no_main_pid(self):
        self.assertFalse(self.stalled(self.stats(age=999), pid="0", uptime=None))

    def test_stats_stopped_updating(self):
        self.assertTrue(self.stalled(self.stats(age=120)))

    def test_no_samples_in_last_minute(self):
        self.assertTrue(self.stalled(self.stats(samples=0)))

    def test_unreadable_stats(self):
        self.assertTrue(self.stalled(stats_error=OSError("gone")))

    def test_malformed_stats(self):
        self.assertTrue(self.stalled({"now": self.NOW}))


class SdrRecoveryTests(unittest.TestCase):
    def setUp(self):
        patches = {
            "get_gpio": mock.patch.object(a.GpioController, "get_gpio", return_value=True),
            "cycle": mock.patch.object(a, "cycle_sdr_rail", return_value=True),
            "wait": mock.patch.object(a, "wait_for_rtlsdr", return_value=True),
            "in_use": mock.patch.object(a, "rtlsdr_in_use", return_value=False),
            "stalled": mock.patch.object(a, "readsb_stalled", return_value=False),
            "call": mock.patch.object(a.subprocess, "call", return_value=0),
            "env": mock.patch.dict(a.os.environ, {}, clear=False),
            "print": mock.patch("builtins.print"),
        }
        self.m = types.SimpleNamespace()
        for name, patcher in patches.items():
            setattr(self.m, name, patcher.start())
            self.addCleanup(patcher.stop)
        a.os.environ.pop("SERVICE_RESULT", None)

    def test_rail_switched_off_is_left_alone(self):
        self.m.get_gpio.return_value = False
        a.os.environ["SERVICE_RESULT"] = "exit-code"
        self.m.stalled.return_value = True
        for stage in ("prestart", "poststop", "check"):
            self.assertEqual(a.sdr_recovery(stage), 0)
        self.m.cycle.assert_not_called()
        self.m.wait.assert_not_called()
        self.m.call.assert_not_called()
        self.m.get_gpio.assert_called_with(a.GPIO_MAP["SDR"])

    def test_prestart_waits_for_high_speed(self):
        self.assertEqual(a.sdr_recovery("prestart"), 0)
        self.m.wait.assert_called_once_with(20, a.rtlsdr_high_speed)
        self.m.cycle.assert_not_called()

    def test_prestart_never_fails_the_unit(self):
        self.m.wait.return_value = False
        self.assertEqual(a.sdr_recovery("prestart"), 0)

    def test_poststop_after_clean_stop_does_nothing(self):
        for result in (None, "success"):
            if result:
                a.os.environ["SERVICE_RESULT"] = result
            self.assertEqual(a.sdr_recovery("poststop"), 0)
        self.m.cycle.assert_not_called()

    def test_poststop_after_failure_power_cycles(self):
        for result in ("exit-code", "signal", "core-dump", "timeout"):
            a.os.environ["SERVICE_RESULT"] = result
            a.sdr_recovery("poststop")
        self.assertEqual(self.m.cycle.call_count, 4)

    def test_poststop_spares_an_sdr_held_by_another_program(self):
        a.os.environ["SERVICE_RESULT"] = "exit-code"
        self.m.in_use.return_value = True
        self.assertEqual(a.sdr_recovery("poststop"), 0)
        self.m.cycle.assert_not_called()

    def test_poststop_survives_a_failed_cycle(self):
        a.os.environ["SERVICE_RESULT"] = "exit-code"
        self.m.cycle.return_value = False
        self.assertEqual(a.sdr_recovery("poststop"), 0)

    def test_check_healthy_does_nothing(self):
        self.assertEqual(a.sdr_recovery("check"), 0)
        self.m.call.assert_not_called()
        self.m.cycle.assert_not_called()

    def test_check_stalled_stops_cycles_then_starts(self):
        self.m.stalled.return_value = True
        order = mock.Mock()
        order.attach_mock(self.m.call, "call")
        order.attach_mock(self.m.cycle, "cycle")
        self.assertEqual(a.sdr_recovery("check"), 0)
        self.assertEqual(order.mock_calls, [
            mock.call.call(["systemctl", "stop", "readsb"]),
            mock.call.cycle(),
            mock.call.call(["systemctl", "start", "readsb"]),
        ])


class UnitFileTests(unittest.TestCase):
    def test_dropins_run_recovery_before_bias_tee(self):
        # systemd applies drop-ins in filename order.
        self.assertEqual(os.path.dirname(a.SDR_RECOVERY_DROPIN_PATH),
                         os.path.dirname(a.SDR_BIASTEE_DROPIN_PATH))
        self.assertLess(os.path.basename(a.SDR_RECOVERY_DROPIN_PATH),
                        os.path.basename(a.SDR_BIASTEE_DROPIN_PATH))
        self.assertTrue(a.SDR_RECOVERY_DROPIN_PATH.startswith(
            "/etc/systemd/system/readsb.service.d/"))

    def test_recovery_dropin(self):
        unit = a.SDR_RECOVERY_DROPIN
        self.assertIn("After=aiov2-rails-boot.service", unit)
        # "+" runs the hooks as root despite readsb's User=readsb.
        self.assertIn("ExecStartPre=+/usr/bin/python3 /usr/local/bin/aiov2_ctl --sdr-recovery prestart", unit)
        self.assertIn("ExecStopPost=+/usr/bin/python3 /usr/local/bin/aiov2_ctl --sdr-recovery poststop", unit)

    def test_watchdog_units(self):
        self.assertIn("ExecStart=/usr/bin/python3 /usr/local/bin/aiov2_ctl --sdr-recovery check",
                      a.SDR_WATCHDOG_SERVICE_UNIT)
        self.assertIn("Type=oneshot", a.SDR_WATCHDOG_SERVICE_UNIT)
        self.assertIn("OnUnitActiveSec=1min", a.SDR_WATCHDOG_TIMER_UNIT)
        self.assertIn("WantedBy=timers.target", a.SDR_WATCHDOG_TIMER_UNIT)
        self.assertEqual(a.SDR_WATCHDOG_TIMER.rsplit(".", 1)[0],
                         a.SDR_WATCHDOG_SERVICE.rsplit(".", 1)[0])

    def test_biastee_dropin(self):
        self.assertIn(f"ExecStartPre=+{a.RTL_BIAST} -d 0 -b 1", a.SDR_BIASTEE_DROPIN)


class SdrBiasteeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dropin = os.path.join(tmp.name, "readsb.service.d", "20-aiov2-sdr-biastee.conf")
        self.biast = os.path.join(tmp.name, "rtl_biast")
        open(self.biast, "w").close()

        self.readsb_active = True
        self.calls = []

        def call(cmd, **kwargs):
            self.calls.append(cmd)
            if cmd[:3] == ["systemctl", "is-active", "--quiet"]:
                return 0 if self.readsb_active else 3
            return 0

        for patcher in (
            mock.patch.object(a, "SDR_BIASTEE_DROPIN_PATH", self.dropin),
            mock.patch.object(a, "RTL_BIAST", self.biast),
            mock.patch.object(a, "SDR_BIASTEE_DROPIN", "[Service]\nExecStartPre=+rtl_biast -b 1\n"),
            mock.patch.object(a.os, "geteuid", return_value=0),
            mock.patch.object(a.subprocess, "call", side_effect=call),
            mock.patch.object(a, "rtlsdr_present", return_value=True),
            mock.patch.object(a, "rtlsdr_in_use", return_value=False),
            mock.patch("builtins.print"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def actions(self):
        return [c for c in self.calls if c[:2] != ["systemctl", "is-active"]]

    def test_status(self):
        with mock.patch("builtins.print") as out:
            a.sdr_biastee("status")
            os.makedirs(os.path.dirname(self.dropin))
            open(self.dropin, "w").close()
            a.sdr_biastee("status")
        self.assertEqual([c.args[0] for c in out.call_args_list],
                         ["SDR bias tee for readsb: OFF", "SDR bias tee for readsb: ON"])
        self.assertEqual(self.calls, [])

    def test_on_writes_dropin_and_restarts_running_readsb(self):
        self.assertEqual(a.sdr_biastee("on"), 0)
        with open(self.dropin) as f:
            self.assertEqual(f.read(), a.SDR_BIASTEE_DROPIN)
        self.assertEqual(os.stat(self.dropin).st_mode & 0o777, 0o644)
        self.assertEqual(self.actions(), [
            ["systemctl", "daemon-reload"],
            ["systemctl", "restart", "readsb"],
        ])

    def test_on_with_readsb_stopped_only_writes_dropin(self):
        self.readsb_active = False
        self.assertEqual(a.sdr_biastee("on"), 0)
        self.assertTrue(os.path.exists(self.dropin))
        self.assertEqual(self.actions(), [["systemctl", "daemon-reload"]])

    def test_on_without_rtl_biast_refuses(self):
        os.remove(self.biast)
        self.assertEqual(a.sdr_biastee("on"), 1)
        self.assertFalse(os.path.exists(self.dropin))
        self.assertEqual(self.calls, [])

    def test_off_with_readsb_running_frees_tuner_to_switch_off(self):
        a.sdr_biastee("on")
        self.calls.clear()
        self.assertEqual(a.sdr_biastee("off"), 0)
        self.assertFalse(os.path.exists(self.dropin))
        self.assertEqual(self.actions(), [
            ["systemctl", "daemon-reload"],
            ["systemctl", "stop", "readsb"],
            [self.biast, "-d", "0", "-b", "0"],
            ["systemctl", "start", "readsb"],
        ])

    def test_off_with_readsb_stopped_switches_idle_tuner_off(self):
        self.readsb_active = False
        self.assertEqual(a.sdr_biastee("off"), 0)
        self.assertEqual(self.actions(), [
            ["systemctl", "daemon-reload"],
            [self.biast, "-d", "0", "-b", "0"],
        ])

    def test_off_leaves_a_tuner_held_by_another_program(self):
        self.readsb_active = False
        with mock.patch.object(a, "rtlsdr_in_use", return_value=True):
            a.sdr_biastee("off")
        self.assertEqual(self.actions(), [["systemctl", "daemon-reload"]])

    def test_off_without_rtl_biast_does_not_crash(self):
        os.remove(self.biast)
        for active in (True, False):
            self.readsb_active = active
            self.calls.clear()
            self.assertEqual(a.sdr_biastee("off"), 0)
            self.assertNotIn(self.biast, [c[0] for c in self.calls])

    def test_non_root_reruns_with_sudo(self):
        with mock.patch.object(a.os, "geteuid", return_value=1000), \
                mock.patch.object(a, "rerun_with_sudo", side_effect=SystemExit) as rerun:
            with self.assertRaises(SystemExit):
                a.sdr_biastee("on")
        rerun.assert_called_once_with(["--sdr-biastee", "on"])
        self.assertFalse(os.path.exists(self.dropin))


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, os.path.join(REPO, "aiov2_ctl.py"), *args],
                              capture_output=True, text=True, timeout=30)

    def test_sdr_recovery_rejects_unknown_stage(self):
        for args in (["--sdr-recovery"], ["--sdr-recovery", "bogus"]):
            res = self.run_cli(*args)
            self.assertEqual(res.returncode, 1)
            self.assertIn("Usage: aiov2_ctl --sdr-recovery prestart|poststop|check", res.stdout)

    def test_sdr_biastee_rejects_unknown_state(self):
        for args in (["--sdr-biastee"], ["--sdr-biastee", "maybe"]):
            res = self.run_cli(*args)
            self.assertEqual(res.returncode, 1)
            self.assertIn("Usage: aiov2_ctl --sdr-biastee on|off|status", res.stdout)

    def test_help_and_completion_list_sdr_biastee(self):
        self.assertIn("--sdr-biastee", a.HELP_TEXT)
        self.assertIn("--sdr-biastee", a.BASH_COMPLETION)


class PygpsclientProfileTests(unittest.TestCase):
    INSTALLER_LINE = 'export PATH="/home/"$(logname)"/.pygpsclient/bin:$PATH"'

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = tmp.name
        for patcher in (
            mock.patch.object(a, "target_home", return_value=self.home),
            mock.patch("builtins.print"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def write(self, name, text):
        with open(os.path.join(self.home, name), "w") as f:
            f.write(text)

    def read(self, name):
        with open(os.path.join(self.home, name)) as f:
            return f.read()

    def test_rewrites_logname_path_to_home_and_dedupes(self):
        self.write(".zprofile", "\n".join([
            'if [ -f "$HOME/.profile" ]; then',
            "    emulate sh -c '. \"$HOME/.profile\"'",
            "fi",
            "# Path to PyGPSClient executable",
            self.INSTALLER_LINE,
            "# Path to PyGPSClient executable",
            self.INSTALLER_LINE,
            "",
        ]))
        a.prepare_pygpsclient_profile()
        text = self.read(".zprofile")
        self.assertNotIn("logname", text)
        self.assertEqual(text.count(a.PYGPSCLIENT_PATH_LINE), 1)
        self.assertEqual(text.count("# Path to PyGPSClient executable"), 1)
        self.assertIn("[ -n \"$ZSH_VERSION\" ] && emulate sh -c", text)

    def test_keeps_indentation(self):
        self.write(".zshrc", "if true; then\n  " + self.INSTALLER_LINE + "\nfi\n")
        a.prepare_pygpsclient_profile()
        self.assertIn("\n  " + a.PYGPSCLIENT_PATH_LINE + "\n", self.read(".zshrc"))

    def test_is_idempotent(self):
        self.write(".zprofile", "# Path to PyGPSClient executable\n" + self.INSTALLER_LINE + "\n")
        a.prepare_pygpsclient_profile()
        first = self.read(".zprofile")
        mtime = os.stat(os.path.join(self.home, ".zprofile")).st_mtime_ns
        a.prepare_pygpsclient_profile()
        self.assertEqual(self.read(".zprofile"), first)
        self.assertEqual(os.stat(os.path.join(self.home, ".zprofile")).st_mtime_ns, mtime)

    def test_unrelated_profile_untouched(self):
        original = 'export PATH="$HOME/bin:$PATH"\nalias ll="ls -l"\n'
        self.write(".zprofile", original)
        a.prepare_pygpsclient_profile()
        self.assertEqual(self.read(".zprofile"), original)

    def test_path_line_expands_to_the_venv(self):
        out = subprocess.run(
            ["sh", "-c", a.PYGPSCLIENT_PATH_LINE + '; echo "$PATH"'],
            env={"HOME": "/home/someone", "PATH": "/usr/bin"},
            capture_output=True, text=True,
        ).stdout.strip()
        self.assertEqual(out, "/home/someone/.pygpsclient/bin:/usr/bin")


class FinalizePygpsclientTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = tmp.name
        self.venv = os.path.join(self.home, ".pygpsclient")
        os.makedirs(os.path.join(self.venv, "bin"))
        self.launcher = os.path.join(self.venv, "bin", "pygpsclient")
        open(self.launcher, "w").close()
        self.config = os.path.join(self.home, "pygpsclient.json")

        self.uid = os.getuid()
        self.owner = types.SimpleNamespace(pw_uid=self.uid, pw_gid=os.getgid())
        self.link_exists = True
        real_exists = os.path.exists

        def exists(path):
            if path == "/usr/local/bin/pygpsclient":
                return self.link_exists
            return real_exists(path)

        self.m = types.SimpleNamespace()
        for name, patcher in {
            "user": mock.patch.object(a, "target_user", return_value="kali"),
            "home": mock.patch.object(a, "target_home", return_value=self.home),
            "pwd": mock.patch.object(a.pwd, "getpwnam", side_effect=lambda u: self.owner),
            "exists": mock.patch.object(a.os.path, "exists", side_effect=exists),
            "symlink": mock.patch.object(a.os, "symlink"),
            "chown": mock.patch.object(a.os, "chown"),
            "call": mock.patch.object(a.subprocess, "call", return_value=0),
            "print": mock.patch("builtins.print"),
        }.items():
            setattr(self.m, name, patcher.start())
            self.addCleanup(patcher.stop)

    def test_writes_gps_config_owned_by_user(self):
        a.finalize_pygpsclient()
        with open(self.config) as f:
            config = json.load(f)
        self.assertEqual(config, {"userport_s": "/dev/serial0", "bpsrate_n": 9600})
        self.m.chown.assert_called_once_with(self.config, self.owner.pw_uid, self.owner.pw_gid)
        self.m.home.assert_called_with("kali")

    def test_keeps_existing_config(self):
        with open(self.config, "w") as f:
            f.write('{"userport_s": "/dev/ttyUSB0"}')
        a.finalize_pygpsclient()
        with open(self.config) as f:
            self.assertEqual(json.load(f), {"userport_s": "/dev/ttyUSB0"})
        self.m.chown.assert_not_called()

    def test_hands_root_owned_venv_back_to_user(self):
        self.owner = types.SimpleNamespace(pw_uid=self.uid + 1, pw_gid=4242)
        a.finalize_pygpsclient()
        self.m.call.assert_called_once_with(["chown", "-R", f"{self.uid + 1}:4242", self.venv])

    def test_user_owned_venv_left_alone(self):
        a.finalize_pygpsclient()
        self.m.call.assert_not_called()

    def test_creates_missing_launcher(self):
        self.link_exists = False
        a.finalize_pygpsclient()
        self.m.symlink.assert_called_once_with(self.launcher, "/usr/local/bin/pygpsclient")

    def test_existing_launcher_kept_and_setup_still_runs(self):
        a.finalize_pygpsclient()
        self.m.symlink.assert_not_called()
        self.assertTrue(os.path.exists(self.config))

    def test_launcher_failure_does_not_stop_setup(self):
        self.link_exists = False
        self.m.symlink.side_effect = PermissionError("read-only")
        a.finalize_pygpsclient()
        self.assertTrue(os.path.exists(self.config))

    def test_nothing_without_installed_venv(self):
        os.remove(self.launcher)
        a.finalize_pygpsclient()
        self.assertFalse(os.path.exists(self.config))
        self.m.call.assert_not_called()
        self.m.symlink.assert_not_called()

    def test_nothing_without_a_home(self):
        self.m.home.return_value = None
        a.finalize_pygpsclient()
        self.assertFalse(os.path.exists(self.config))

    def test_second_run_is_a_no_op(self):
        a.finalize_pygpsclient()
        self.m.chown.reset_mock()
        a.finalize_pygpsclient()
        self.m.chown.assert_not_called()
        self.m.call.assert_not_called()

    def test_tips_point_at_the_gps_uart(self):
        self.assertIn("/dev/serial0", a.POST_INSTALL_TIPS)
        self.assertIn("--boot-rail GPS on", a.POST_INSTALL_TIPS)
        self.assertNotIn("second serial device", a.POST_INSTALL_TIPS)


if __name__ == "__main__":
    unittest.main()
