@echo off
chcp 65001

call runtime\python.exe webui_cpu.py

@echo 请按任意键继续
call pause