# CEFIELD Stripe Setup Guide

To start charging customers for API calls and AI diagnostics, follow these steps. Half of the process has been automated by the script in this repository.

## 1. Local Setup (Your Machine)
Since your API runs locally on Docker during development, Stripe needs a secure tunnel to send Webhooks (payment alerts) to your PC.

1. **Install Stripe CLI**:
   - Windows (PowerShell): `scoop install stripe`
   - MacOS: `brew install stripe/stripe-cli/stripe`
   - Linux: `sudo apt-get install stripe`
2. **Login to Stripe**:
   Open a terminal and type:
   ```bash
   stripe login
   ```
   *Press Enter to open your browser and authorize.*
3. **Start the Webhook Tunnel**:
   Keep this terminal open to forward real-time payment data to your local FastAPI:
   ```bash
   stripe listen --forward-to localhost:8000/webhooks/stripe
   ```
   *Note the `whsec_...` secret it prints out!*

## 2. Configure Environment Variables
Create a file named `.env` in the root of your repository and fill it with your keys from the Stripe Dashboard:
```bash
STRIPE_API_KEY=sk_test_... (From Stripe Dashboard)
STRIPE_WEBHOOK_SECRET=whsec_... (From the terminal output above)
ANTHROPIC_API_KEY=sk-ant-... (Your Claude AI Key)
```

## 3. Run the Automated Product Setup (AI Generated)
Instead of clicking through the Stripe Dashboard to set up Metered Billing, simply run our automated script. It will create the Product and the "Pay-per-API-call" meter:

```bash
export STRIPE_API_KEY=sk_test_your_key_here
python scripts/setup_stripe.py
```
*Output will show: ✅ Product created, ✅ Meter created.*

## 4. Test the System!
1. Start your server: `docker-compose up --build`
2. Onboard a test customer using your FastAPI Swagger UI (`http://localhost:8000/docs#/default/onboard_customer_api_v1_billing_onboard_post`).
3. Send a measurement using your edge agent.
4. Check your Stripe Dashboard under **Billing -> Usage**. You will see `1 API call` billed automatically to the new customer!
