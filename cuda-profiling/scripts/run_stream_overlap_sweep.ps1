$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Exe = Join-Path $ProjectRoot "build\stream_overlap_benchmark.exe"
$ResultsDir = Join-Path $ProjectRoot "results"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ResultFile = Join-Path $ResultsDir "stream_overlap_sweep_$Timestamp.csv"
$PlotFile = Join-Path $ResultsDir "stream_overlap_sweep_$Timestamp.png"

$ChunksList = @(2, 4, 8, 16)
$ItersList = @(16, 64, 256)
$StreamList = @(1, 2, 4, 8)
$Size = 16777216
$Repeat = 5
$BlockSize = 256

if (-not (Test-Path $Exe)) {
    throw "Stream overlap benchmark executable not found. Run scripts\build.ps1 first."
}

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

$Header = "method,size,total_mb,chunks,streams,chunk_elems,iters,repeat,total_ms,effective_transfer_gb_s,speedup_vs_pageable_seq,speedup_vs_pinned_seq,max_abs_error,async_engine_count,concurrent_kernels,status"
$Header | Out-File -FilePath $ResultFile -Encoding utf8

foreach ($iters in $ItersList) {
    foreach ($chunks in $ChunksList) {
        $validStreams = @($StreamList | Where-Object { $_ -le $chunks })
        $invalidStreams = @($StreamList | Where-Object { $_ -gt $chunks })
        $streamCounts = $validStreams -join ","

        Write-Host "Running stream overlap sweep: chunks=$chunks, streams=$streamCounts, iters=$iters"
        $lines = & $Exe --size $Size --chunks $chunks --repeat $Repeat --block-size $BlockSize --iters $iters --stream-counts $streamCounts

        foreach ($line in $lines) {
            if ([string]::IsNullOrWhiteSpace($line)) {
                continue
            }

            if ($line.StartsWith("method,")) {
                continue
            }

            if ($line.Contains(",")) {
                "$line,ok" | Out-File -FilePath $ResultFile -Encoding utf8 -Append
            }
        }

        foreach ($streams in $invalidStreams) {
            $totalMb = ($Size * 4.0) / (1024.0 * 1024.0)
            $chunkElems = [Math]::Floor($Size / $chunks)
            "pinned_streams_$streams,$Size,$('{0:F6}' -f $totalMb),$chunks,$streams,$chunkElems,$iters,$Repeat,,,,,,,,invalid_streams_gt_chunks" |
                Out-File -FilePath $ResultFile -Encoding utf8 -Append
        }
    }
}

Add-Type -AssemblyName System.Drawing

$rows = Import-Csv $ResultFile | Where-Object {
    $_.status -eq "ok" -and $_.method -like "pinned_streams_*"
}

$width = 1600
$height = 620
$marginLeft = 76
$marginRight = 34
$marginTop = 82
$marginBottom = 86
$panelGap = 42
$itersValues = @($ItersList)
$chunkValues = @($ChunksList)
$streamValues = @($StreamList)
$panelWidth = [Math]::Floor(($width - $marginLeft - $marginRight - ($panelGap * ($itersValues.Count - 1))) / $itersValues.Count)
$panelHeight = $height - $marginTop - $marginBottom
$maxMs = ($rows | ForEach-Object { [double]$_.total_ms } | Measure-Object -Maximum).Maximum
$maxMs = [Math]::Ceiling($maxMs * 1.10)

$bitmap = New-Object System.Drawing.Bitmap $width, $height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.Clear([System.Drawing.Color]::White)

$titleFont = New-Object System.Drawing.Font "Arial", 20, ([System.Drawing.FontStyle]::Bold)
$labelFont = New-Object System.Drawing.Font "Arial", 10
$smallFont = New-Object System.Drawing.Font "Arial", 9
$axisPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(80, 80, 80)), 1
$gridPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(220, 220, 220)), 1
$brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(30, 30, 30))
$colors = @{
    1 = [System.Drawing.Color]::FromArgb(76, 120, 168)
    2 = [System.Drawing.Color]::FromArgb(245, 133, 24)
    4 = [System.Drawing.Color]::FromArgb(84, 162, 75)
    8 = [System.Drawing.Color]::FromArgb(178, 121, 162)
}

$graphics.DrawString("CUDA stream overlap sweep", $titleFont, $brush, 24, 22)

for ($panelIndex = 0; $panelIndex -lt $itersValues.Count; $panelIndex++) {
    $iters = $itersValues[$panelIndex]
    $x0 = $marginLeft + ($panelIndex * ($panelWidth + $panelGap))
    $y0 = $marginTop
    $y1 = $marginTop + $panelHeight

    $graphics.DrawString("iters=$iters", $labelFont, $brush, $x0, 58)

    for ($tick = 0; $tick -le 4; $tick++) {
        $value = $maxMs * $tick / 4.0
        $y = $y1 - ($panelHeight * $tick / 4.0)
        $graphics.DrawLine($gridPen, $x0, $y, $x0 + $panelWidth, $y)
        if ($panelIndex -eq 0) {
            $graphics.DrawString(("{0:F0}" -f $value), $smallFont, $brush, 28, $y - 8)
        }
    }

    $graphics.DrawLine($axisPen, $x0, $y0, $x0, $y1)
    $graphics.DrawLine($axisPen, $x0, $y1, $x0 + $panelWidth, $y1)

    foreach ($chunks in $chunkValues) {
        $x = $x0 + (($chunks - $chunkValues[0]) / ($chunkValues[-1] - $chunkValues[0])) * $panelWidth
        $graphics.DrawString("$chunks", $smallFont, $brush, $x - 8, $y1 + 10)
    }

    foreach ($streams in $streamValues) {
        $pen = New-Object System.Drawing.Pen $colors[$streams], 3
        $pointBrush = New-Object System.Drawing.SolidBrush $colors[$streams]
        $previous = $null

        foreach ($chunks in $chunkValues) {
            $match = @($rows | Where-Object {
                [int]$_.iters -eq $iters -and [int]$_.chunks -eq $chunks -and [int]$_.streams -eq $streams
            })

            if ($match.Count -eq 0) {
                $previous = $null
                continue
            }

            $ms = [double]$match[0].total_ms
            $x = $x0 + (($chunks - $chunkValues[0]) / ($chunkValues[-1] - $chunkValues[0])) * $panelWidth
            $y = $y1 - (($ms / $maxMs) * $panelHeight)

            if ($null -ne $previous) {
                $graphics.DrawLine($pen, $previous.X, $previous.Y, $x, $y)
            }

            $graphics.FillEllipse($pointBrush, $x - 4, $y - 4, 8, 8)
            $previous = [PSCustomObject]@{ X = $x; Y = $y }
        }

        $pen.Dispose()
        $pointBrush.Dispose()
    }
}

$graphics.DrawString("total time (ms)", $labelFont, $brush, 12, $marginTop + 4)
$graphics.DrawString("chunks", $labelFont, $brush, [Math]::Floor($width / 2) - 20, $height - 36)

$legendX = $width - 230
$legendY = 24
foreach ($streams in $streamValues) {
    $legendPen = New-Object System.Drawing.Pen $colors[$streams], 4
    $graphics.DrawLine($legendPen, $legendX, $legendY + 8, $legendX + 34, $legendY + 8)
    $graphics.DrawString("$streams streams", $labelFont, $brush, $legendX + 42, $legendY)
    $legendY += 22
    $legendPen.Dispose()
}

$bitmap.Save($PlotFile, [System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose()
$bitmap.Dispose()

Write-Host ""
Write-Host "Saved results to $ResultFile"
Write-Host "Saved plot to $PlotFile"
