use std::fs;
use std::io::{self, Write};
use std::path::Path;
use hf_hub::repository::AddSource;

fn ensure_token(token: &str) {
    if !token.is_empty() {
        std::env::set_var("HF_TOKEN", token);
    }
}

fn parse_repo_id(repo_url: &str) -> Result<(String, String), String> {
    let r = parse_repo(repo_url);
    let (owner, name) = r
        .split_once('/')
        .ok_or_else(|| format!("repo_id inválido: {}", repo_url))?;
    Ok((owner.to_string(), name.to_string()))
}

pub fn get_hf_token() -> String {
    if let Ok(t) = std::env::var("HF_TOKEN") {
        if !t.is_empty() { return t; }
    }
    print!("HuggingFace token (Enter para público): ");
    io::stdout().flush().unwrap();
    let mut token = String::new();
    io::stdin().read_line(&mut token).unwrap();
    let t = token.trim().to_string();
    if !t.is_empty() {
        std::env::set_var("HF_TOKEN", &t);
    }
    t
}

pub fn upload_file(
    repo_url: &str,
    local_path: &str,
    path_in_repo: &str,
    token: &str,
    commit_msg: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    ensure_token(token);
    let (owner, name) = parse_repo_id(repo_url)?;
    let client = hf_hub::HFClientSync::new()?;
    client
        .model(&owner, &name)
        .upload_file()
        .source(AddSource::file(std::path::PathBuf::from(local_path)))
        .path_in_repo(path_in_repo)
        .commit_message(commit_msg)
        .send()?;
    println!("  Subido: {} → {}/{}", local_path, name, path_in_repo);
    Ok(())
}

pub fn download_file(
    repo_url: &str,
    filename: &str,
    token: &str,
    dest_dir: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    fs::create_dir_all(dest_dir)?;
    let dest_path = Path::new(dest_dir).join(filename);
    if dest_path.exists() {
        println!("  Ya existe: {}", dest_path.display());
        return Ok(dest_path.to_string_lossy().to_string());
    }

    ensure_token(token);
    let (owner, name) = parse_repo_id(repo_url)?;
    let client = hf_hub::HFClientSync::new()?;
    let path = client
        .model(&owner, &name)
        .download_file()
        .filename(filename)
        .send()?;

    fs::copy(&path, &dest_path)?;
    println!(
        "  Descargado: {} ({} bytes)",
        dest_path.display(),
        fs::metadata(&dest_path)?.len()
    );
    Ok(dest_path.to_string_lossy().to_string())
}

pub fn parse_repo(url: &str) -> String {
    url.trim_end_matches('/')
        .strip_prefix("https://huggingface.co/")
        .or_else(|| url.strip_prefix("http://huggingface.co/"))
        .or_else(|| url.strip_prefix("hf.co/"))
        .unwrap_or(url.trim_end_matches('/'))
        .to_string()
}
