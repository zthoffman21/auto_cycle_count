[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string]$ImageName = "cycle-count-vision:latest"
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

Write-Host "Building $ImageName..."
Invoke-Docker -DockerArguments @("build", "--target", "vision", "--tag", $ImageName, $workspace)
Write-Host "Build complete." -ForegroundColor Green
