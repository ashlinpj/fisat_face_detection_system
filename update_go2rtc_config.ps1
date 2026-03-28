param(
    [Parameter(Mandatory = $true)]
    [string]$SourceUrl
)

$configPath = Join-Path $PSScriptRoot "config.py"
if (-not (Test-Path $configPath)) {
    Write-Error "config.py not found at $configPath"
    exit 1
}

$content = Get-Content $configPath -Raw

# Keep setup idempotent and safe: update only known go2rtc/RTSP lines.
$content = $content -replace '(?m)^USE_RTSP\s*=\s*False\s*(#.*)?$', 'USE_RTSP = True  # Use RTSP stream instead of laptop webcam'
$content = $content -replace '(?m)^GO2RTC_STREAM_NAME\s*=\s*.*$', 'GO2RTC_STREAM_NAME = "cam"'
$content = $content -replace '(?m)^GO2RTC_SOURCE_URL\s*=\s*.*$', ('GO2RTC_SOURCE_URL = "' + $SourceUrl + '"  # Original camera stream ingested by go2rtc')
$content = $content -replace '(?m)^RTSP_URL\s*=\s*.*$', 'RTSP_URL = f"rtsp://127.0.0.1:8554/{GO2RTC_STREAM_NAME}"  # App reads local low-latency go2rtc relay'

Set-Content -Path $configPath -Value $content -Encoding UTF8
Write-Host "config.py updated successfully."
exit 0
