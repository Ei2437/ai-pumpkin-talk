# push.ps1
$DIR = Get-Location

Write-Host "[Git] Working in $DIR..."
Set-Location $DIR
Write-Host "[Git] Staging changes for personal repo..."
git add -A

# コミットするか確認
$changes = git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "[Git] No changes to commit in personal repo."
} else {
    $msg = Read-Host "Commit Message (personal)"
    if ([string]::IsNullOrWhiteSpace($msg)) {
        $msg = "Auto Update $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    }
    git commit -m "$msg"
}

Write-Host "[Git] Pushing to personal origin..."
git push origin main

Write-Host "[Git] Done."
