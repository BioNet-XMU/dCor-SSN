"""Peak resident memory on the current platform, in MiB."""
import ctypes
import sys


def peak_rss_mib():
    if sys.platform == "win32":
        class ProcessCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        process = ctypes.windll.kernel32.GetCurrentProcess()
        counters = ProcessCounters()
        counters.cb = ctypes.sizeof(counters)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(process), ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise ctypes.WinError()
        return counters.PeakWorkingSetSize / 2**20
    import resource
    units = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * units / 2**20
