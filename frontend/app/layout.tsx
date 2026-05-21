import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NemoPilot — AI project builder",
  description:
    "Describe your software idea, connect GitHub, and let Nemotron agents plan, build, and export a full project repository.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
