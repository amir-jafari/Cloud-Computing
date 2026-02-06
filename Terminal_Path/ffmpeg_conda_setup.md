# Adding Existing FFmpeg Installation to Conda Environment

If you have ffmpeg installed on Windows and it works in CMD/PowerShell but not in your Anaconda/PyCharm terminal, follow these steps to add it to your conda environment's PATH.

## Step 1: Find where ffmpeg is installed

Open regular CMD (not PyCharm) and run:

```cmd
where ffmpeg
```

This will show you the path, likely something like `C:\ffmpeg\bin\ffmpeg.exe`

## Step 2: Add it to your conda environment's PATH

In PyCharm's terminal (or Anaconda terminal):

```powershell
conda env config vars set PATH="C:\ffmpeg\bin;$PATH"
```

**Important:** Replace `C:\ffmpeg\bin` with your actual path from Step 1.

## Step 3: Reactivate your environment

```powershell
conda deactivate
conda activate base
```

Replace `base` with your environment name if you're using a different environment.

## Step 4: Verify installation

```powershell
ffmpeg -version
```

You should now see the ffmpeg version information.

---

## Why this is needed

PyCharm's terminal uses the conda environment's PATH, which is isolated from your Windows system PATH. That's why ffmpeg works in CMD but not in PyCharm/Anaconda terminals.

## Alternative: Install via conda

If you prefer a cleaner solution, you can install ffmpeg directly through conda:

```powershell
conda install -c conda-forge ffmpeg
```

This eliminates the need for PATH configuration.
