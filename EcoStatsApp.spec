# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['run_webview.py'],
    pathex=['.'],
    binaries=[],
    datas=[('templates', 'templates'), ('static', 'static')],
    hiddenimports=['waitress'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    runtime_tmpdir=None,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='EcoStatsApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# For macOS: Create an app bundle
app_bundle = BUNDLE(
    exe,
    name='EcoStatsApp.app',
    icon=None,
    bundle_identifier=None,
) 