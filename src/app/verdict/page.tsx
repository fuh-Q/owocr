"use client";

import styles from "../../styling/verdict.module.scss";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

import { emit, listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/tauri";

type PredictionPayload = { prediction: string; confidence: number };

function Verdict() {
    const [prediction, setPred] = useState<string | null>(null);
    const [confidence, setConf] = useState<number | null>(null);

    listen<PredictionPayload>("prediction", ({ payload }) => {
        setPred(payload.prediction);
        setConf(payload.confidence);
    });

    emit("ready");

    useEffect(() => {
        setTimeout(() => {
            const butt = document.getElementById("close-button") as HTMLButtonElement | null;
            if (!butt) {
                return;
            }

            butt.disabled = false;
            butt.addEventListener("click", () => invoke("close_pred_windows"));
        }, 3000);
    }, [prediction]);

    if (!prediction || !confidence) {
        return null;
    }

    return (
        <div className={"site-body " + styles["site-body"]} style={{ marginTop: 0 }}>
            <span>Prediction</span>
            <div className={styles.prediction}>{prediction}</div>
            <span className={styles.confidence}>confidence: {Math.round(confidence * 100)}%</span>
            <button id="close-button" className={styles.butt} disabled>
                Close
            </button>
        </div>
    );
}

export default dynamic(async () => Verdict, { ssr: false });
