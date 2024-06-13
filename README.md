<div style="font-size: xx-large;">Ow<strong>OCR</strong></div>

---

This is my not-so-successful attempt at creating an OCR

It ships with various limitions including:

-   Can only scan one character at a time
-   Not very accurate

---

Requirements for self-deployment:

-   NodeJS (v20 and up recommended)
-   Rustc (v1.7.3 and up recommended)

Steps:

-   Run the following commands:

```sh
git clone https://github.com/fuh-Q/owocr
cd owocr
npm i
npm run tauri dev
```

You may also choose to build a release deployment that is optimized for production. This will require a re-compilation of the app (takes a while but this is normal for Rust). The command to do so is

```sh
npm run tauri build
npm run tauri start
```
