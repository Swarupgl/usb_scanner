# Why Raspberry Pi / Embedded “USB Sanitizer” (even if the model can run on a PC)

Your sir’s question is valid:

> “If your model works like Defender, why not install it on every computer directly?”

Here’s the clean technical answer.

---

## 1) Different goal: endpoint AV vs. *USB gate* / pre-entry control

- **Endpoint AV** (Defender) runs *inside* the PC you want to protect.
- Your project is a **pre-entry USB gate**: it checks removable media **before** it touches important PCs.

This matters because “where the scanning happens” changes the security guarantees.

---

## 2) Trust boundary: scan on a separate device so the host stays clean

If you scan on the same Windows PC you are protecting:

- The USB is already plugged into that PC.
- Any OS-level parser bugs, auto-run behaviors, thumbnail parsing, shortcut tricks, etc. can be triggered **before** your app gets a chance to scan.

With a Raspberry Pi kiosk/gateway:

- Untrusted USB is handled by the Pi first.
- You can mount read-only / noexec policies (common in embedded deployments).
- Even if something goes wrong, you compromise the **gate**, not the office PC.

This is the same principle as “scan email attachments at the mail gateway, not only on the user’s laptop.”

---

## 3) Deployment reality: you often can’t install custom security on every PC

In labs, schools, offices, and hackathon venues:

- You may not have admin rights to install drivers/services.
- Policies may forbid disabling/altering existing AV.
- Users might uninstall/stop your software.

A Pi-based kiosk is:

- **Plug-and-play** and independent.
- One device can serve many PCs.
- No changes needed on the protected PCs.

---

## 4) Consistency + audit: one controlled gate gives repeatable behavior

On many PCs, you get:

- Different AV versions, different settings, different exclusions.
- Different CPU/RAM, so scanning rules may differ.

On one embedded gate, you get:

- Same model, same thresholds, same quarantine policy.
- Central logs and a clear “clean / quarantined” outcome.

That’s easier to defend in evaluation.

---

## 5) Performance & UX: kiosk = predictable, low-noise, fast

- Real-time scanning on endpoints can slow users or create conflicts with existing AV.
- A kiosk UI can enforce a simple workflow: “Insert USB → Wait → Take sanitized files”.

---

## 6) Honest claim to make to your sir

Say it like this:

- “Yes sir, the model can run on a PC too. But our project is intentionally an **external sanitizer / gateway** so the protected PC never touches untrusted USB files until they pass the gate. It’s a different deployment model and gives a stronger trust boundary.”

---

## Optional: how to answer follow-up questions

**Q: So can you still run it on Windows directly?**

- “Yes. The scanner is Python and can run on Windows. We chose Pi for a kiosk + hardening + one-device-for-many-computers deployment.”

**Q: Is it ‘better than Defender’?**

- “We don’t claim to replace Defender. We add a *different control*: a deterministic structure-integrity gate + ML score before files ever reach the endpoint.”
