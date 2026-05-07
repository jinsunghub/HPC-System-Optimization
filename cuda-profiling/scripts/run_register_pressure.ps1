$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Exe = Join-Path $ProjectRoot "build\register_pressure_benchmark.exe"
$ResultsDir = Join-Path $ProjectRoot "results"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ResultFile = Join-Path $ResultsDir "register_pressure_$Timestamp.csv"
$PlotFile = Join-Path $ResultsDir "register_pressure_$Timestamp.png"

if (-not (Test-Path $Exe)) {
    throw "Register pressure benchmark executable not found. Run scripts\build.ps1 first."
}

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

& $Exe --size 4194304 --repeat 20 --block-size 256 --iters 256 |
    Tee-Object -FilePath $ResultFile

Add-Type -AssemblyName System.Drawing

$rows = @(Import-Csv $ResultFile | Sort-Object {[int]$_.pressure})
$maxMs = ($rows | ForEach-Object { [double]$_.kernel_ms } | Measure-Object -Maximum).Maximum
$maxYLeft = [Math]::Ceiling($maxMs * 1.15 * 10.0) / 10.0

$width = 1500
$height = 620
$marginLeft = 82
$marginRight = 72
$marginTop = 80
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
$timePen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(76, 120, 168)), 3
$occPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(245, 133, 24)), 3
$timeBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(76, 120, 168))
$occBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(245, 133, 24))

function X-ForPressure($pressure) {
    $minLog = [Math]::Log10(4.0)
    $maxLog = [Math]::Log10(96.0)
    return $marginLeft + (([Math]::Log10([double]$pressure) - $minLog) / ($maxLog - $minLog)) * $panelWidth
}

function Y-Left($value) {
    return $marginTop + $panelHeight - (($value / $maxYLeft) * $panelHeight)
}

function Y-Occupancy($value) {
    return $marginTop + $panelHeight - ($value * $panelHeight)
}

function Draw-Series($field, $pen, $pointBrush, $mapper) {
    $previous = $null
    foreach ($row in $rows) {
        $x = X-ForPressure ([int]$row.pressure)
        $value = [double]$row.$field
        $y = & $mapper $value
        if ($null -ne $previous) {
            $graphics.DrawLine($pen, $previous.X, $previous.Y, $x, $y)
        }
        $graphics.FillEllipse($pointBrush, $x - 4, $y - 4, 8, 8)
        $previous = [PSCustomObject]@{ X = $x; Y = $y }
    }
}

$graphics.DrawString("Register pressure and occupancy sweep", $titleFont, $brush, 24, 22)
$graphics.DrawString("kernel time (ms)", $labelFont, $brush, 12, $marginTop + 4)
$graphics.DrawString("theoretical occupancy", $labelFont, $brush, $width - 166, $marginTop + 4)

$x0 = $marginLeft
$y0 = $marginTop
$y1 = $marginTop + $panelHeight

for ($tick = 0; $tick -le 5; $tick++) {
    $value = $maxYLeft * $tick / 5.0
    $y = $y1 - ($panelHeight * $tick / 5.0)
    $graphics.DrawLine($gridPen, $x0, $y, $x0 + $panelWidth, $y)
    $graphics.DrawString(("{0:F1}" -f $value), $smallFont, $brush, 28, $y - 8)

    $occValue = $tick / 5.0
    $graphics.DrawString(("{0:F1}" -f $occValue), $smallFont, $brush, $width - 62, $y - 8)
}

$graphics.DrawLine($axisPen, $x0, $y0, $x0, $y1)
$graphics.DrawLine($axisPen, $x0, $y1, $x0 + $panelWidth, $y1)
$graphics.DrawLine($axisPen, $x0 + $panelWidth, $y0, $x0 + $panelWidth, $y1)

foreach ($pressure in @(4, 8, 16, 32, 64, 96)) {
    $x = X-ForPressure $pressure
    $graphics.DrawString("$pressure", $smallFont, $brush, $x - 8, $y1 + 10)
}

Draw-Series "kernel_ms" $timePen $timeBrush ${function:Y-Left}
Draw-Series "theoretical_occupancy" $occPen $occBrush ${function:Y-Occupancy}

$graphics.DrawString("pressure level, log scale", $labelFont, $brush, [Math]::Floor($width / 2) - 70, $height - 38)

$legendX = $width - 330
$legendY = 26
$graphics.DrawLine($timePen, $legendX, $legendY + 8, $legendX + 34, $legendY + 8)
$graphics.DrawString("kernel time", $labelFont, $brush, $legendX + 42, $legendY)
$legendY += 24
$graphics.DrawLine($occPen, $legendX, $legendY + 8, $legendX + 34, $legendY + 8)
$graphics.DrawString("theoretical occupancy", $labelFont, $brush, $legendX + 42, $legendY)

$bitmap.Save($PlotFile, [System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose()
$bitmap.Dispose()

Write-Host ""
Write-Host "Saved results to $ResultFile"
Write-Host "Saved plot to $PlotFile"
