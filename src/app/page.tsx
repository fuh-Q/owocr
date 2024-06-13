"use client";

import styles from "../styling/page.module.scss";
import Footer from "../components/Footer";

import dynamic from "next/dynamic";

import { invoke } from "@tauri-apps/api/tauri";
import { open } from "@tauri-apps/api/dialog";

async function clickHandler() {
    const imgFilter = {
        name: "Image",
        extensions: ["jpeg", "jpg", "png", "bmp", "ico"],
    };

    const selected = await open({
        multiple: false,
        filters: [imgFilter],
    });

    if (typeof selected !== "string") {
        return;
    }

    await invoke("close_pred_windows");
    let prediction, confidence;

    try {
        const opts = { filename: selected, showImage: true };
        [prediction, confidence] = await invoke<[string, number]>("predict", opts);
    } catch (e: any) {
        const field = document.getElementById("error-reporting") as HTMLDivElement | null;
        if (!field) {
            return;
        }

        field.classList.add("error-blob");
        field.textContent = `error: ${e}`;

        return;
    }

    dismissError();
    await invoke("verdict", { prediction, confidence });
}

function dismissError() {
    const field = document.getElementById("error-reporting") as HTMLDivElement | null;
    if (!field) {
        return;
    }

    field.classList.remove("error-blob");
    field.textContent = "";
}

function Home() {
    return (
        <>
            <div className="site-body">
                <h1 className={styles.heading}>
                    Ow<span className="brighten">OCR</span>
                </h1>
                <h5 className={styles.subheading}>
                    A <span className="brighten">character classifier</span>. Kinda
                </h5>
                <button className={styles.butt} onClick={clickHandler}>
                    Upload Image
                </button>
                <div id="error-reporting" onClick={dismissError}></div>
                <div className="warning-blob">
                    <strong>Limitations</strong>
                    <br />
                    <br /> • Cannot predict on lowercase letters
                    <br /> • Can only predict on one character at a time
                    <br /> • May yield inaccurate results
                </div>
                <Footer />
            </div>
            <img className={styles.ferris} width="96" height="64" src="/ferris.png"></img>
        </>
    );
}

export default dynamic(async () => Home, { ssr: false });
