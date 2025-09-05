# Ensure script runs as administrator
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator"))
{
    Write-Host "Please run this script as Administrator." -ForegroundColor Red
    exit 1
}

# Update WinGet sources
Write-Host "Updating WinGet sources..."
winget source update

function Install-App {
    param(
        [string]$Id,
        [string]$Name
    )

    # Check if app is installed
    $installed = winget list --id $Id -e | Select-String $Id

    if ($installed) {
        Write-Host "$Name is already installed. Skipping..." -ForegroundColor Cyan
    }
    else {
        Write-Host "Installing $Name..."
        winget install --id $Id -e --source winget --accept-source-agreements --accept-package-agreements
    }
}

# Install Git
Install-App -Id "Git.Git" -Name "Git"

# Install Blender
Install-App -Id "BlenderFoundation.Blender" -Name "Blender"

# Install Miniconda
Install-App -Id "Anaconda.Miniconda3" -Name "Miniconda"

# Find blender.exe in common locations (Program Files, LocalAppData, and a user Apps folder)
$searchRoots = @(
  "$env:ProgramFiles\Blender Foundation",
  "$env:LOCALAPPDATA",
  "$env:USERPROFILE\Apps"
) | Where-Object { Test-Path $_ }

$blenderExe = $null
foreach ($root in $searchRoots) {
  try {
    $hit = Get-ChildItem -Path $root -Filter blender.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($hit) { $blenderExe = $hit; break }
  } catch {}
}

if (-not $blenderExe) {
  Write-Host "Couldn't find blender.exe automatically." -ForegroundColor Yellow
  Write-Host "If you installed a portable ZIP, point PATH to the folder that contains blender.exe."
  Write-Host "Example: C:\Users\<you>\Apps\Blender-4.4.0\blender-4.4.0-windows-x64"
  return
}

$blenderDir = $blenderExe.DirectoryName
Write-Host "Found Blender at: $blenderDir"

# Add to USER PATH if not already present
$userPath = [Environment]::GetEnvironmentVariable("Path","User")
if ($userPath -notmatch [Regex]::Escape($blenderDir)) {
  [Environment]::SetEnvironmentVariable("Path", ($userPath.TrimEnd(';') + ';' + $blenderDir), "User")
  Write-Host "Added to USER PATH: $blenderDir" -ForegroundColor Green
} else {
  Write-Host "Blender directory already in USER PATH." -ForegroundColor Cyan
}

# Make it available in the current PowerShell session immediately
if ($env:Path -notmatch [Regex]::Escape($blenderDir)) {
  $env:Path += ";" + $blenderDir
}

# Refresh PATH environment variable
Write-Host "Refreshing PATH environment variables..."
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "`nAll done!"
Write-Host "Close and reopen your terminal or log out/in to ensure PATH changes are fully applied." -ForegroundColor Yellow
