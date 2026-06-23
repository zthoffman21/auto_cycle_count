[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,

    [ValidateNotNullOrEmpty()]
    [string]$ImageName = "cycle-count-vision:latest",

    [ValidateNotNullOrEmpty()]
    [string]$ContainerName = "cycle-count-vision",

    [string]$ModelDirectory = "",

    [string]$RfdetrCheckpoint = "rf-detr-seg-small-totes.pth",

    [string]$Dinov2ModelDirectory = "dinov2-vits14",

    [string]$Dinov2Classifier = "empty-cell-head.safetensors",

    [string]$PatchAnomalyModel = "patch-anomaly.safetensors",

    [string]$GroundingDinoModel = "grounding-dino-base",

    [ValidateRange(0.0, 1.0)]
    [double]$GroundingDinoBoxThreshold = 0.25,

    [ValidateRange(0, 600)]
    [int]$HealthTimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Docker {
    param(
        [Parameter(Mandatory)]
        [string[]]$DockerArguments,

        [switch]$DiscardOutput
    )

    # Windows PowerShell turns Docker's stderr progress into exceptions under Stop
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($DiscardOutput) {
            & docker @DockerArguments *> $null
        }
        else {
            & docker @DockerArguments
        }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    if ($exitCode -ne 0) {
        throw "docker $($DockerArguments -join ' ') failed with exit code $exitCode."
    }
}

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$artifactDirectory = Join-Path $workspace "data\artifacts"
$trainingDirectory = Join-Path $workspace "data\training"
New-Item -ItemType Directory -Force -Path $artifactDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $trainingDirectory | Out-Null
if (-not $ModelDirectory) {
    $ModelDirectory = Join-Path $workspace "models"
}

$requiredArtifacts = @(
    (Join-Path $ModelDirectory $RfdetrCheckpoint),
    (Join-Path $ModelDirectory $Dinov2ModelDirectory),
    (Join-Path $ModelDirectory $Dinov2Classifier)
)
foreach ($artifact in $requiredArtifacts) {
    if (-not (Test-Path $artifact)) {
        throw "Required model artifact is missing: $artifact"
    }
}
$ModelDirectory = (Resolve-Path $ModelDirectory).Path

try {
    Invoke-Docker -DockerArguments @("info") -DiscardOutput
}
catch {
    throw "Docker daemon is unavailable. Start Docker Desktop and retry."
}

$existingContainer = Invoke-Docker -DockerArguments @(
    "ps", "--all", "--quiet", "--filter", "name=^/$ContainerName$"
)
if ($existingContainer) {
    $containerImageId = Invoke-Docker -DockerArguments @(
        "inspect", "--format", "{{.Image}}", $ContainerName
    )
    $currentImageId = Invoke-Docker -DockerArguments @(
        "inspect", "--format", "{{.Id}}", $ImageName
    )
    if ($containerImageId -ne $currentImageId) {
        Write-Host "Container '$ContainerName' is using a stale image. Recreating..." -ForegroundColor Yellow
        Invoke-Docker -DockerArguments @("rm", "--force", $ContainerName) -DiscardOutput
    }
    else {
        throw "Container '$ContainerName' already exists and is up to date. Remove it with 'docker rm $ContainerName' before starting a new one."
    }
}

Write-Host "Starting $ContainerName on port $Port..."
$artifactMount = "type=bind,source=$artifactDirectory,target=/app/data/artifacts"
$trainingMount = "type=bind,source=$trainingDirectory,target=/app/data/training"
$modelMount = "type=bind,source=$ModelDirectory,target=/models,readonly"
$runArguments = @(
    "run",
    "--detach",
    "--name", $ContainerName,
    "--publish", "${Port}:8000",
    "--gpus", "all",
    "--mount", $artifactMount,
    "--mount", $trainingMount,
    "--mount", $modelMount,
    "--env", "CYCLE_COUNT_RFDETR_CHECKPOINT_PATH=/models/$RfdetrCheckpoint",
    "--env", "CYCLE_COUNT_DINOV2_MODEL_PATH=/models/$Dinov2ModelDirectory",
    "--env", "CYCLE_COUNT_DINOV2_CLASSIFIER_PATH=/models/$Dinov2Classifier",
    "--env", "CYCLE_COUNT_VISION_DEVICE=cuda"
)
$patchAnomalyPath = Join-Path $ModelDirectory $PatchAnomalyModel
if (Test-Path $patchAnomalyPath) {
    $runArguments += "--env", "CYCLE_COUNT_PATCH_ANOMALY_MODEL_PATH=/models/$PatchAnomalyModel"
    Write-Host "Patch anomaly model found."
}
$groundingDinoPath = Join-Path $ModelDirectory $GroundingDinoModel
if (Test-Path $groundingDinoPath) {
    $runArguments += "--env", "CYCLE_COUNT_GROUNDING_DINO_MODEL_PATH=/models/$GroundingDinoModel"
    $runArguments += "--env", "CYCLE_COUNT_GROUNDING_DINO_BOX_THRESHOLD=$GroundingDinoBoxThreshold"
    Write-Host "GroundingDINO model found (box_threshold=$GroundingDinoBoxThreshold)."
}
$runArguments += $ImageName
Invoke-Docker -DockerArguments $runArguments | Out-Null

$healthUri = "http://localhost:$Port/health/live"
$dashboardUri = "http://localhost:$Port/"

try {
    $healthy = $false
    $healthDeadline = [DateTime]::UtcNow.AddSeconds($HealthTimeoutSeconds)
    while ([DateTime]::UtcNow -lt $healthDeadline) {
        try {
            $response = Invoke-RestMethod -Uri $healthUri -TimeoutSec 2
            if ($response.status -eq "ok") {
                $healthy = $true
                break
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }

    if (-not $healthy) {
        throw "Health check did not pass within $HealthTimeoutSeconds seconds."
    }
}
catch {
    Write-Host "Container logs:" -ForegroundColor Yellow
    Invoke-Docker -DockerArguments @("logs", $ContainerName)
    Invoke-Docker -DockerArguments @("stop", $ContainerName) | Out-Null
    throw
}

Write-Host "Cycle Count Vision is running." -ForegroundColor Green
Write-Host "Dashboard: $dashboardUri"
Write-Host "API docs:  http://localhost:$Port/docs"
Write-Host "Stop with:  docker stop $ContainerName"
Write-Host "Remove with: docker rm $ContainerName"
