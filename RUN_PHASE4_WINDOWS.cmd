@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" goto :usage
python tools\run_phase4_windows.py %*
exit /b %ERRORLEVEL%

:usage
echo Usage: RUN_PHASE4_WINDOWS.cmd TARGET
echo.
echo Main targets:
echo   verify-raw
echo   controlled
echo   home-preprocess
echo   home-downstream
echo   electronics-preprocess
echo   electronics-downstream
echo   amazon-both-from-raw
echo   home-figure-data
echo   electronics-figures
echo   electronics-reporting
echo   cross-category-table
echo   reporting
echo   final-check
echo   reproduce-final-from-raw
echo.
echo Home resume targets:
echo   resume-after-home-preprocess
echo   resume-after-home-reference
echo   resume-after-home-reuse
echo   resume-after-home-shape
echo   resume-after-home-reference-history
echo.
echo Electronics-only resume targets ^(do NOT return to Home^):
echo   resume-after-electronics-preprocess
echo   resume-after-electronics-reference
echo   resume-after-electronics-primary-attack
echo   resume-after-electronics-primary-reuse
echo   resume-after-electronics-matched-twins
echo   resume-after-electronics-shape
echo   resume-after-electronics-complementarity
echo   resume-after-electronics-reference-history
echo   resume-after-electronics-strength
echo   resume-after-electronics-k6-audit
echo   resume-after-electronics-full-background
echo   resume-after-electronics-self-influence
exit /b 2
