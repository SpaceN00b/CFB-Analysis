# PyInstaller spec for the Windows app: pyinstaller packaging/cfb_app.spec --noconfirm
# Output: dist/CFB Predictor/ (one folder with CFB Predictor.exe and CFB Engine.exe; the installer in
# packaging/installer.iss wraps it).
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent

datas = []
datas += collect_data_files("streamlit")      # front-end static files
datas += copy_metadata("streamlit")            # streamlit reads its own version at start
datas += [
    (str(ROOT / "src" / "cfb" / "ui" / "app.py"), "cfb/ui"),     # Streamlit runs a script path
    (str(ROOT / "dashboard" / "config.toml"), "dashboard"),     # run config fallback
    # The report's dashboard template: the local engine renders reports (cfb.report.html).
    (str(ROOT / "src" / "cfb" / "report" / "templates"), "cfb/report/templates"),
]

hiddenimports = collect_submodules("streamlit") + collect_submodules("cfb")

a = Analysis(
    [str(ROOT / "packaging" / "cfb_app.py")],
    pathex=[str(ROOT / "src")],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CFB Predictor",
    console=True,  # the window shows the server log; closing it stops the app
    icon=None,
)
# The same program without a console window: Task Scheduler and the app's Run buttons start the
# local engine with it (`"CFB Engine.exe" engine all`), so no window opens. Output goes to the
# engine's log files.
engine_exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CFB Engine",
    console=False,
    icon=None,
)
coll = COLLECT(exe, engine_exe, a.binaries, a.datas, name="CFB Predictor")
