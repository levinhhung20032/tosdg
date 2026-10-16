# Brings the console window that started the studio (cmd, or Windows Terminal hosting it) to the front.
# Windows refuses SetForegroundWindow from a background process, so a synthetic Alt key press is sent first (the usual workaround).
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class Front {
  [DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow();
  [DllImport("user32.dll")] public static extern IntPtr GetAncestor(IntPtr h, uint f);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr h);
  [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int cx, int cy, uint f);
  [DllImport("user32.dll")] public static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);
}
'@
$h = [Front]::GetConsoleWindow()
if ($h -eq [IntPtr]::Zero) { exit 0 }
$root = [Front]::GetAncestor($h, 3)                          # GA_ROOTOWNER: the Windows Terminal window when the console is a pseudo-console
if ($root -ne [IntPtr]::Zero) { $h = $root }
if ([Front]::IsIconic($h)) { [void][Front]::ShowWindow($h, 9) }   # SW_RESTORE
[Front]::keybd_event(0x12, 0, 0, [UIntPtr]::Zero)             # Alt down
[Front]::keybd_event(0x12, 0, 2, [UIntPtr]::Zero)             # Alt up
[void][Front]::SetForegroundWindow($h)
[void][Front]::BringWindowToTop($h)
# flash it on top once, then drop the always-on-top flag again (HWND_TOPMOST = -1, HWND_NOTOPMOST = -2; NOMOVE|NOSIZE = 3)
[void][Front]::SetWindowPos($h, [IntPtr](-1), 0, 0, 0, 0, 3)
[void][Front]::SetWindowPos($h, [IntPtr](-2), 0, 0, 0, 0, 3)
