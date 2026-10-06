//! `ollaya update`: install the latest release over this one (#41).
//!
//! The release is found the way the install scripts find it: GitHub redirects
//! `/releases/latest` to `/releases/tag/v<version>`, so no API call and no rate limit. Updating
//! re-runs the platform's install script into this installation's prefix, so the script's checks
//! (sha256, GPU pack, the systemd service) apply unchanged. Installs the script does not own (the
//! macOS app, the Windows desktop app, a container) get the right instruction instead.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use ollaya_server::config::VERSION;

const REPO: &str = "ollaya-dev/ollaya";
const INSTALL_SH: &str = "https://ollaya.dev/install.sh";
const INSTALL_PS1: &str = "https://ollaya.dev/install.ps1";
const DOWNLOAD: &str = "https://ollaya.dev/download";

/// `major.minor.patch`, ignoring any pre-release suffix.
fn parse(v: &str) -> Option<(u64, u64, u64)> {
    let v = v.trim().trim_start_matches('v');
    let core = v.split(['-', '+']).next()?;
    let mut it = core.split('.').map(|p| p.parse::<u64>().ok());
    Some((it.next()??, it.next()??, it.next()??))
}

/// The latest release's version, from the redirect of `/releases/latest`.
async fn latest() -> Result<String> {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .user_agent(format!("ollaya/{VERSION}"))
        .build()?;
    let url = format!("https://github.com/{REPO}/releases/latest");
    let resp = client
        .head(&url)
        .send()
        .await
        .with_context(|| format!("checking {url}"))?;
    let location = resp
        .headers()
        .get(reqwest::header::LOCATION)
        .and_then(|l| l.to_str().ok())
        .with_context(|| format!("{url} did not redirect to a release ({})", resp.status()))?;
    let tag = location
        .rsplit_once("/releases/tag/")
        .map(|(_, t)| t)
        .with_context(|| format!("unexpected release location {location}"))?;
    Ok(tag.trim_start_matches('v').to_owned())
}

/// `canonicalize` gives a verbatim path on Windows (`\\?\C:\...`), which PowerShell's `Join-Path`
/// and most programs reject (#50): turn it back into the ordinary form. Other paths are unchanged.
fn plain_path(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    match s.strip_prefix(r"\\?\") {
        Some(rest) => PathBuf::from(rest),
        None => p,
    }
}

/// How this binary was installed.
#[derive(Debug, PartialEq)]
enum Install {
    /// By an install script, under this prefix (`<prefix>/bin/ollaya`).
    Script(PathBuf),
    /// With the desktop app (the macOS app bundle, the Windows installer, an AppImage or a Linux
    /// package), which updates as a whole.
    DesktopApp,
    /// In a container image.
    Container,
}

fn install_kind(exe: &Path) -> Install {
    let s = exe.to_string_lossy();
    // The install scripts put the binary in <prefix>/bin; the desktop app puts it next to the app
    // (Windows), in the bundle (macOS), in the AppImage or in /usr/bin (the .deb and .rpm).
    let in_bin = exe
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|n| n.eq_ignore_ascii_case("bin"));
    if s.contains(".app/Contents/")
        || s.contains("/.mount_")
        || s.starts_with("/usr/bin/")
        || !in_bin
    {
        return Install::DesktopApp;
    }
    if Path::new("/.dockerenv").exists() || Path::new("/run/.containerenv").exists() {
        return Install::Container;
    }
    let prefix = exe
        .parent()
        .and_then(Path::parent)
        .map_or_else(|| PathBuf::from("/usr/local"), Path::to_path_buf);
    Install::Script(prefix)
}

pub async fn update(check: bool) -> Result<()> {
    let latest = latest().await?;
    let (Some(have), Some(want)) = (parse(VERSION), parse(&latest)) else {
        bail!("cannot compare versions {VERSION} and {latest}");
    };
    if want <= have {
        println!("ollaya {VERSION} is the latest version");
        return Ok(());
    }
    println!("ollaya {latest} is available (this is {VERSION})");
    let exe = std::env::current_exe()
        .and_then(|p| p.canonicalize())
        .map(plain_path)
        .context("locating the ollaya executable")?;
    let kind = install_kind(&exe);
    if check {
        println!("Run `ollaya update` to install it.");
        return Ok(());
    }
    match kind {
        Install::DesktopApp => {
            println!(
                "This ollaya came with the Ollaya desktop app: install the new app from {DOWNLOAD}"
            );
            Ok(())
        }
        Install::Container => {
            println!(
                "This ollaya runs in a container: pull the new image, for example \
                 `docker pull ghcr.io/ollaya-dev/ollaya:{latest}`, and recreate the container"
            );
            Ok(())
        }
        Install::Script(prefix) => run_installer(&prefix, &latest),
    }
}

/// Re-run the install script for `version` into `prefix`.
fn run_installer(prefix: &Path, version: &str) -> Result<()> {
    println!("Installing ollaya {version} into {}", prefix.display());
    let status = if cfg!(windows) {
        Command::new("powershell")
            .args([
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                &format!("irm {INSTALL_PS1} | iex"),
            ])
            .env("OLLAYA_VERSION", version)
            .env("OLLAYA_INSTALL_DIR", prefix)
            .status()
    } else {
        Command::new("sh")
            .args(["-c", &format!("curl -fsSL {INSTALL_SH} | sh")])
            .env("OLLAYA_VERSION", version)
            .env("OLLAYA_INSTALL_DIR", prefix)
            .status()
    }
    .context("starting the install script")?;
    if !status.success() {
        bail!("the install script failed ({status})");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions() {
        assert_eq!(parse("0.7.5"), Some((0, 7, 5)));
        assert_eq!(parse("v0.10.0"), Some((0, 10, 0)));
        assert_eq!(parse("1.2.3-rc1"), Some((1, 2, 3)));
        assert!(parse("0.10.0") > parse("0.9.9"));
        assert_eq!(parse("x"), None);
    }

    #[test]
    fn verbatim_paths_become_plain() {
        assert_eq!(
            plain_path(PathBuf::from(
                r"\\?\C:\Users\me\AppData\Local\Programs\Ollaya\bin\ollaya.exe"
            )),
            PathBuf::from(r"C:\Users\me\AppData\Local\Programs\Ollaya\bin\ollaya.exe")
        );
        assert_eq!(
            plain_path(PathBuf::from(r"\\?\UNC\server\share\bin\ollaya.exe")),
            PathBuf::from(r"\\server\share\bin\ollaya.exe")
        );
        assert_eq!(
            plain_path(PathBuf::from("/usr/local/bin/ollaya")),
            PathBuf::from("/usr/local/bin/ollaya")
        );
    }

    #[test]
    fn install_kinds() {
        assert_eq!(
            install_kind(Path::new("/Applications/Ollaya.app/Contents/MacOS/ollaya")),
            Install::DesktopApp
        );
        assert_eq!(
            install_kind(Path::new("/usr/bin/ollaya")),
            Install::DesktopApp
        );
        if !Path::new("/.dockerenv").exists() && !Path::new("/run/.containerenv").exists() {
            assert_eq!(
                install_kind(Path::new("/home/me/.local/bin/ollaya")),
                Install::Script(PathBuf::from("/home/me/.local"))
            );
        }
    }
}
