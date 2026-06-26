[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string]$ImageName = "cycle-count-vision:latest",

    [string]$DependencyImage = "",

    [switch]$BuildDependencyImage
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

try {
    Invoke-Docker -DockerArguments @("info") -DiscardOutput
}
catch {
    throw "Docker daemon is unavailable. Start Docker Desktop and retry."
}

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$defaultDependencyImage = "cycle-count-vision-dependencies:latest"

if ($BuildDependencyImage -and -not $DependencyImage) {
    $DependencyImage = $defaultDependencyImage
}

if ($BuildDependencyImage) {
    Write-Host "Building vision dependency image $DependencyImage..."
    Invoke-Docker -DockerArguments @(
        "build",
        "--target", "vision-dependencies",
        "--tag", $DependencyImage,
        $workspace
    )
}
elseif (-not $DependencyImage) {
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & docker image inspect $defaultDependencyImage *> $null
        $dependencyImageExists = $LASTEXITCODE -eq 0
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($dependencyImageExists) {
        $DependencyImage = $defaultDependencyImage
    }
}

Write-Host "Building $ImageName..."
$buildArguments = @("build", "--target", "vision", "--tag", $ImageName)
if ($DependencyImage) {
    Write-Host "Using vision dependency image $DependencyImage."
    $buildArguments += "--build-arg", "VISION_DEPENDENCIES_IMAGE=$DependencyImage"
}
$buildArguments += $workspace
Invoke-Docker -DockerArguments $buildArguments
Write-Host "Build complete." -ForegroundColor Green
