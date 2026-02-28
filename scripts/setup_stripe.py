import os
import stripe

# Fetch API key from environment
stripe.api_key = os.environ.get("STRIPE_API_KEY")

def setup_stripe():
    if not stripe.api_key or "mock" in stripe.api_key:
        print("ERROR: Please set a valid real STRIPE_API_KEY in your environment.")
        return

    print("Authenticating with Stripe...")
    
    # 1. Create the Meter for Usage-Based Billing
    print("Creating Meter Event 'api_requests'...")
    try:
        meter = stripe.billing.Meter.create(
            display_name="CEFIELD API Calls",
            event_name="api_requests",
            default_aggregation={"formula": "sum"},
        )
        print(f"✅ Meter created successfully: {meter.id}")
    except Exception as e:
        print(f"⚠️ Meter creation info: {e}")

    # 2. Create the Product
    print("Creating Product 'CEFIELD Enterprise Data Access'...")
    product = stripe.Product.create(
        name="CEFIELD Enterprise Data Access",
        description="Access to the Global Resonator Genome & AI Diagnostics."
    )
    print(f"✅ Product created: {product.id}")

    # 3. Create the Usage-Based Price (e.g., 0.05 EUR per API Call)
    print("Creating Price (0.05 EUR per unit)...")
    price = stripe.Price.create(
        product=product.id,
        currency="eur",
        recurring={"usage_type": "metered"},
        unit_amount=5, # 5 cents
        billing_scheme="per_unit",
    )
    print(f"✅ Price created: {price.id}")
    print("\n🎉 Stripe Setup Complete! You are ready to generate real revenue.")

if __name__ == "__main__":
    setup_stripe()
