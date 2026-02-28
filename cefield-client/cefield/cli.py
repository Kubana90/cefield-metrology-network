import click
import pandas as pd
from .node import CefieldNode

@click.group()
def main():
    """CEFIELD Global Resonator Genome - Edge Client CLI"""
    pass

@main.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--f0', required=True, type=float, help="Estimated resonance frequency (Hz)")
@click.option('--hardware', default="Generic_CSV", help="Hardware identifier (e.g. PicoScope_6000)")
@click.option('--node', default="CLI_User", help="Your Lab/Node ID")
@click.option('--url', default="http://localhost:8000", help="CEFIELD Cloud Core URL")
def analyze(filepath, f0, hardware, node, url):
    """
    Process a local CSV file (Time, Voltage) and stream the signature to the Global Brain.
    """
    click.secho(f"[*] Reading raw data from {filepath}...", fg="cyan")
    try:
        # Assuming simple CSV with two columns, no header needed if strict, but let's be robust
        df = pd.read_csv(filepath, header=None)
        if df.shape[1] < 2:
            click.secho("[!] CSV must contain at least two columns: Time and Voltage", fg="red")
            return
            
        time_arr = df.iloc[:, 0].values
        volt_arr = df.iloc[:, 1].values
        
    except Exception as e:
        click.secho(f"[!] Failed to read CSV: {e}", fg="red")
        return

    click.secho(f"[*] Initializing Edge Compute for Node: {node}", fg="cyan")
    client = CefieldNode(node_id=node, api_url=url)
    
    click.secho("[*] Extracting 128-dim physical signature (Hilbert transform)...", fg="yellow")
    result = client.analyze_and_stream(
        time_array=time_arr,
        voltage_array=volt_arr,
        hardware_type=hardware,
        estimated_f0=f0
    )
    
    if result.get("status") == "error":
        click.secho(f"[X] {result.get('message')}", fg="red")
    else:
        click.secho("\n[+] Successfully synced with CEFIELD Global Brain.", fg="green")
        if "alert" in result:
            click.secho(f"\n[!] AI ALERT: {result['alert']}", fg="red", bold=True)
            click.secho(f"Claude 3.5 Diagnostic: {result.get('ai_diagnostic')}", fg="yellow")
        else:
            click.secho(f"\n[+] Status: {result.get('message')}", fg="green")

if __name__ == "__main__":
    main()
