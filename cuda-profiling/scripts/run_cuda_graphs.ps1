$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Exe = Join-Path $ProjectRoot "build\cuda_graphs_benchmark.exe"
$ResultsDir = Join-Path $ProjectRoot "results"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ResultFile = Join-Path $ResultsDir "cuda_graphs_$Timestamp.csv"
$PlotFile = Join-Path $ResultsDir "cuda_graphs_$Timestamp.png"

if (-not (Test-Path $Exe)) {
    throw "CUDA Graphs benchmark executable not found. Run scripts\build.ps1 first."
}

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

& $Exe --sizes 1024,4096,16384,65536,262144,1048576 --repeat 1000 --block-size 256 --iters 1 |
    Tee-Object -FilePath $ResultFile

Add-Type -AssemblyName System.Drawing

$rows = Import-Csv $ResultFile
$plotRows = @($rows | Where-Object { $_.workload -eq "kernel_only" })
$normalRows = @($plotRows | Where-Object { $_.method -eq "normal_launch" })
$graphRows = @($plotRows | Where-Object { $_.method -eq "cuda_graph_replay" })
$sizes = @($normalRows | ForEach-Object { [int64]$_.size })
$maxWall = ($plotRows | ForEach-Object { [double]$_.per_iter_wall_us } | Measure-Object -Maximum).Maximum
$maxEnqueue = ($plotRows | ForEach-Object { [double]$_.per_iter_enqueue_us } | Measure-Object -Maximum).Maximum
$maxY = [Math]::Ceiling([Math]::Max($maxWall, $maxEnqueue) * 1.15)

$width = 1500
$height = 620
$marginLeft = 82
$marginRight = 36
$marginTop = 78
$marginBottom = 92
$panelGap = 60
$panelWidth = [Math]::Floor(($width - $marginLeft - $marginRight - $panelGap) / 2)
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
$normalPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(76, 120, 168)), 3
$graphPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(245, 133, 24)), 3
$normalBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(76, 120, 168))
$graphBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(245, 133, 24))

function Draw-Panel($x0, $title, $field) {
    $y0 = $marginTop
    $y1 = $marginTop + $panelHeight
    $graphics.DrawString($title, $labelFont, $brush, $x0, 54)

    for ($tick = 0; $tick -le 4; $tick++) {
        $value = $maxY * $tick / 4.0
        $y = $y1 - ($panelHeight * $tick / 4.0)
        $graphics.DrawLine($gridPen, $x0, $y, $x0 + $panelWidth, $y)
        $graphics.DrawString(("{0:F1}" -f $value), $smallFont, $brush, 28, $y - 8)
    }

    $graphics.DrawLine($axisPen, $x0, $y0, $x0, $y1)
    $graphics.DrawLine($axisPen, $x0, $y1, $x0 + $panelWidth, $y1)

    $minLog = [Math]::Log10([double]$sizes[0])
    $maxLog = [Math]::Log10([double]$sizes[-1])

    foreach ($size in $sizes) {
        $x = $x0 + (([Math]::Log10([double]$size) - $minLog) / ($maxLog - $minLog)) * $panelWidth
        $graphics.DrawString("$size", $smallFont, $brush, $x - 24, $y1 + 10)
    }

    foreach ($series in @(
        @{ Rows = $normalRows; Pen = $normalPen; Brush = $normalBrush },
        @{ Rows = $graphRows; Pen = $graphPen; Brush = $graphBrush }
    )) {
        $previous = $null
        foreach ($row in $series.Rows) {
            $size = [int64]$row.size
            $value = [double]$row.$field
            $x = $x0 + (([Math]::Log10([double]$size) - $minLog) / ($maxLog - $minLog)) * $panelWidth
            $y = $y1 - (($value / $maxY) * $panelHeight)
            if ($null -ne $previous) {
                $graphics.DrawLine($series.Pen, $previous.X, $previous.Y, $x, $y)
            }
            $graphics.FillEllipse($series.Brush, $x - 4, $y - 4, 8, 8)
            $previous = [PSCustomObject]@{ X = $x; Y = $y }
        }
    }
}

$graphics.DrawString("CUDA Graphs kernel launch overhead", $titleFont, $brush, 24, 22)
Draw-Panel $marginLeft "End-to-end per iteration (us)" "per_iter_wall_us"
Draw-Panel ($marginLeft + $panelWidth + $panelGap) "CPU enqueue per iteration (us)" "per_iter_enqueue_us"
$graphics.DrawString("input elements, log scale", $labelFont, $brush, [Math]::Floor($width / 2) - 60, $height - 38)

$legendX = $width - 260
$legendY = 24
$graphics.DrawLine($normalPen, $legendX, $legendY + 8, $legendX + 34, $legendY + 8)
$graphics.DrawString("normal launch", $labelFont, $brush, $legendX + 42, $legendY)
$legendY += 22
$graphics.DrawLine($graphPen, $legendX, $legendY + 8, $legendX + 34, $legendY + 8)
$graphics.DrawString("CUDA graph replay", $labelFont, $brush, $legendX + 42, $legendY)

$bitmap.Save($PlotFile, [System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose()
$bitmap.Dispose()

Write-Host ""
Write-Host "Saved results to $ResultFile"
Write-Host "Saved plot to $PlotFile"
