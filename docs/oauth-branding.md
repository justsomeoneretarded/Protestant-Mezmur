# Fix the Audience > Publish App Branding Check

This error appears at the publishing step because Google also validates the
Branding configuration. Finish Branding, then return to Audience. Do not restart
the project or replace working Telegram credentials.

## Publish the App Pages

The repository contains a static app homepage, privacy policy and terms at
docs/index.html, docs/privacy.html and docs/terms.html. The policy describes the
current owner-operated application, including public GitHub artifacts and upload
records. Review it before using it as your app's policy; update it if your data
handling differs. It is not a guarantee of Google approval or legal compliance.

1. Open [repository Pages settings](https://github.com/justsomeoneretarded/Protestant-Mezmur/settings/pages).
2. Under Build and deployment, select **Deploy from a branch**.
3. Select **main** and the **/docs** folder, then Save.
4. Wait for the Pages deployment to finish. Open all three URLs below and confirm
   that the actual pages load without signing in. Publishing can take several minutes.
   Do not submit a URL that still returns 404.

| Field | Value |
| --- | --- |
| App homepage | https://justsomeoneretarded.github.io/Protestant-Mezmur/ |
| Privacy policy | https://justsomeoneretarded.github.io/Protestant-Mezmur/privacy.html |
| Terms of service (optional Branding field) | https://justsomeoneretarded.github.io/Protestant-Mezmur/terms.html |

The site's files are ready, but enabling Pages is a repository-administrator
setting and is not performed by the connected repository editing tools.

## Complete Google Branding

1. Open [Google Auth Platform > Branding](https://console.cloud.google.com/auth/branding)
   and select the **Rehoboth Mezmur** project.
2. App name: **Rehoboth Mezmur**.
3. User support email: select your own Google account email.
4. Fill in homepage, privacy and optional terms URLs from the table, after checking
   that they load publicly.
5. Developer contact information: your email address.
6. In Authorized domains, add **justsomeoneretarded.github.io** if Google requests
   the site domain. Do not enter a full URL or try to verify github.com.
7. Save. Return to **Audience > Publish app** and try again.

If Google requests domain ownership or brand verification, complete the Google
Search Console / Verification Center process shown in your project. Public URLs
alone do not establish ownership and do not guarantee verification. GitHub's
shared domain is not a domain you own; use your own Pages hostname (or an owned
custom domain) as appropriate. Never invent a verification token or mark ownership
as verified when it has not been verified.

Adding app URLs and publishing the OAuth consent configuration is separate from
Google approving sensitive-scope access. Complete any additional checks that
Google requires. A Testing-mode grant remains available for initial testing,
but its refresh token expires after seven days for the YouTube scope.

References: [Google branding and domain requirements](https://developers.google.com/identity/protocols/oauth2/production-readiness/brand-verification),
[GitHub Pages publishing](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site).
