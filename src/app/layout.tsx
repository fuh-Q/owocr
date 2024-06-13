import type { Metadata } from "next";
import "../styling/globals.scss";

export const metadata: Metadata = {
    title: "OwOCR",
    description: "OwOCR Optical Character Recognition",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en" style={{ background: "black" }}>
            <body>{children}</body>
        </html>
    );
}
