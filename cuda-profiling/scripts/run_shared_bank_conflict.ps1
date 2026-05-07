$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Exe = Join-Path $ProjectRoot "build\shared_bank_conflict_benchmark.exe"
$ResultsDir = Join-Path $ProjectRoot "results"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ResultFile = Join-Path $ResultsDir "shared_bank_conflict_$Timestamp.csv"
$PlotFile = Join-Path $ResultsDir "shared_bank_conflict_$Timestamp.png"

if (-not (Test-Path $Exe)) {
    throw "Shared bank conflict benchmark executable not found. Run scripts\build.ps1 first."
}

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

& $Exe --strides 1,2,4,8,16,32 --blocks 256 --block-size 256 --repeat 20 --inner-iters 2048 |
    Tee-Object -FilePath $ResultFile

Add-Type -AssemblyName System.Drawing

$rows = Import-Csv $ResultFile
$conflictRows = @($rows | Where-Object { $_.method -eq "conflict" -or $_.method -eq "conflict_free" } | Sort-Object {[int]$_.requested_stride})
$paddedRows = @($rows | Where-Object { $_.method -eq "padded" } | Sort-Object {[int]$_.requested_stride})
$strides = @($conflictRows | ForEach-Object { [int]$_.requested_stride })
$maxMs = ($rows | ForEach-Object { [double]$_.total_ms } | Measure-Object -Maximum).Maximum
$maxY = [Math]::Ceiling($maxMs * 1.20 * 100.0) / 100.0

$width = 1400
$height = 620
$marginLeft = 78
$marginRight = 34
$marginTop = 78
$marginBottom = 90
$panelWidth = $width - $marginLeft - $marginRight
$panelHeight = $height - $marginTop - $marginBottom

$bitmap = New-Object System.Drawing.Bitmap $width, $height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.Clear([System.Drawing.Color]::White)

$titleFont = New-Object System.Drawing.Font "Arial", 20, ([System.Drawing.FontStyle]::Bold)
$labelFont = New-Object System.Drawing.Font "Arial", 10
$smallFont = New-Object System.Drawing.Font "Arial", 9
$brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(30, 30, 30))
$axisPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(80, 80, 80)), 1
$gridPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(220, 220, 220)), 1
$conflictPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(76, 120, 168)), 3
$paddedPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(245, 133, 24)), 3
$conflictBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(76, 120, 168))
$paddedBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(245, 133, 24))

function X-ForStride($stride) {
    $minLog = [Math]::Log10(1.0)
    $maxLog = [Math]::Log10(32.0)
    return $marginLeft + (([Math]::Log10([double]$stride) - $minLog) / ($maxLog - $minLog)) * $panelWidth
}

function Y-ForMs($ms) {
    return $marginTop + $panelHeight - (($ms / $maxY) * $panelHeight)
}

function Draw-Series($seriesRows, $pen, $pointBrush) {
    $previous = $null
    foreach ($row in $seriesRows) {
        $x = X-ForStride ([int]$row.requested_stride)
        $y = Y-ForMs ([double]$row.total_ms)
        if ($null -ne $previous) {
            $graphics.DrawLine($pen, $previous.X, $previous.Y, $x, $y)
        }
        $graphics.FillEllipse($pointBrush, $x - 4, $y - 4, 8, 8)
        $previous = [PSCustomObject]@{ X = $x; Y = $y }
    }
}

$graphics.DrawString("Shared memory bank conflict", $titleFont, $brush, 24, 22)
$graphics.DrawString("per-launch kernel time (ms)", $labelFont, $brush, 14, $marginTop + 4)

$x0 = $marginLeft
$y0 = $marginTop
$y1 = $marginTop + $panelHeight

for ($tick = 0; $tick -le 5; $tick++) {
    $value = $maxY * $tick / 5.0
    $y = $y1 - ($panelHeight * $tick / 5.0)
    $graphics.DrawLine($gridPen, $x0, $y, $x0 + $panelWidth, $y)
    $graphics.DrawString(("{0:F2}" -f $value), $smallFont, $brush, 28, $y - 8)
}

$graphics.DrawLine($axisPen, $x0, $y0, $x0, $y1)
$graphics.DrawLine($axisPen, $x0, $y1, $x0 + $panelWidth, $y1)

foreach ($stride in @(1, 2, 4, 8, 16, 32)) {
    $x = X-ForStride $stride
    $graphics.DrawString("$stride", $smallFont, $brush, $x - 8, $y1 + 10)
}

Draw-Series $conflictRows $conflictPen $conflictBrush
Draw-Series $paddedRows $paddedPen $paddedBrush

$graphics.DrawString("requested stride, log scale", $labelFont, $brush, [Math]::Floor($width / 2) - 70, $height - 38)

$legendX = $width - 260
$legendY = 26
$graphics.DrawLine($conflictPen, $legendX, $legendY + 8, $legendX + 34, $legendY + 8)
$graphics.DrawString("conflict stride", $labelFont, $brush, $legendX + 42, $legendY)
$legendY += 24
$graphics.DrawLine($paddedPen, $legendX, $legendY + 8, $legendX + 34, $legendY + 8)
$graphics.DrawString("padded stride+1", $labelFont, $brush, $legendX + 42, $legendY)

$bitmap.Save($PlotFile, [System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose()
$bitmap.Dispose()

Write-Host ""
Write-Host "Saved results to $ResultFile"
Write-Host "Saved plot to $PlotFile"
