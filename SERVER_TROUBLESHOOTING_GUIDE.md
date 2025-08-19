# Server Troubleshooting Guide

## Problem: Code Changes Not Reflecting in Frontend

**Root Cause**: Multiple server instances running simultaneously on the same port.

**Symptoms**:
- Code changes don't appear in the frontend
- Server appears to be running normally
- No obvious error messages
- Old responses still coming through

## Solution Steps

### 1. Check for Existing Processes
```bash
# View all Python processes with details
powershell "Get-Process python* | Select-Object ProcessName,Id,CommandLine"

# Quick check for processes using port 8000
netstat -ano | findstr :8000
```

### 2. Kill All Python Processes (Recommended)
```bash
# Nuclear option - kills all Python processes
powershell "Get-Process python* | Stop-Process -Force"
```

### 3. Alternative: Kill Specific Process
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill specific process by PID (replace <PID> with actual process ID)
taskkill /F /PID <PID>
```

## Prevention Best Practices

### 1. Clean Startup Script
Create `start_clean.bat` in project root:
```batch
@echo off
echo Killing any existing Python processes...
powershell "Get-Process python* | Stop-Process -Force" 2>nul
echo Starting fresh server...
cd backend
uv run uvicorn app:app --reload --port 8000
pause
```

### 2. Development Workflow
1. **Before starting server**: Check for existing processes
2. **Always stop cleanly**: Use `Ctrl+C` to stop server
3. **Close terminals properly**: Don't just close terminal window
4. **When debugging**: Kill all processes and start fresh
5. **Branch switching**: Always restart server completely

### 3. IDE Integration Tips
- Use IDE's integrated terminal for better process management
- Don't run multiple development servers simultaneously
- Keep track of which terminals have running processes

### 4. Port Management
```bash
# Use different port if needed
uvicorn app:app --reload --port 8001

# Check all listening ports
netstat -an | findstr LISTENING
```

## Quick Debug Commands

```bash
# Emergency reset - kill everything and restart
powershell "Get-Process python* | Stop-Process -Force"
cd backend && uv run uvicorn app:app --reload --port 8000

# Check if server is actually running
curl http://localhost:8000/api/courses

# View server processes with full command line
wmic process where "name='python.exe'" get processid,commandline
```

## Common Scenarios

### Scenario 1: Server "restarts" but changes don't show
- **Problem**: Old server still running in background
- **Fix**: Kill all Python processes, start fresh

### Scenario 2: Port already in use error
- **Problem**: Previous server didn't shut down cleanly
- **Fix**: Kill process using the port, then restart

### Scenario 3: Multiple terminal sessions
- **Problem**: Started server in multiple terminals
- **Fix**: Close all terminals, kill processes, use single terminal

## Verification Steps

After starting server, verify it's working:
1. Check server logs for startup messages
2. Test a simple API endpoint: `GET /api/courses`
3. Make a small code change and verify it reflects immediately
4. Monitor server logs for reload messages

## Notes

- Windows doesn't always clean up background processes properly
- The `--reload` flag can sometimes leave zombie processes
- Always prefer clean shutdown over closing terminal windows
- When in doubt, kill all Python processes and start fresh