import { invoke } from "@tauri-apps/api/tauri";

export default function Footer() {
    return (
        <div className="site-footer">
            <div className="footer-text">• </div>
            Version {invoke<string>("get_version")}
        </div>
    );
}
