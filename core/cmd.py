#!coding=utf-8
import subprocess

def cmd_exec(cmd_str):
    st = subprocess.STARTUPINFO
    st.dwFlags = subprocess.STARTF_USESHOWWINDOW
    st.wShowWindow = subprocess.SW_HIDE

    p = subprocess.Popen(cmd_str, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while p.poll() is None:
        line = p.stdout.readline()
        line = line.strip()
        if line:
            print('potree: [{}]'.format(line))
    if p.returncode == 0:
        print('ok')
        return True
    return False