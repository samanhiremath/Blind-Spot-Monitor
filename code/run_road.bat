@echo off
cd /d "%~dp0"
REM Stationary radar watching cars pass. Zone: 1 to 5 m to the right, 2 to 8 m forward (edit the numbers to fit your setup).
python live_view.py live --cli COM5 --data COM6 --cfg blindspot_awr1843.cfg --x-min 1 --x-max 3 --y-min 1 --y-max 3 --min-speed 1.5 --min-snr 18 --on-frames 2 --off-frames 5 --view-range 10
echo.
echo Finished. Read any messages above, then press a key to close.
pause
