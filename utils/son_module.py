def recommend_actions(row):
    actions = []
    if row['RSRP'] < -95:
        actions.append("Increase transmit power")
    if row['RSRQ'] < -10:
        actions.append("Adjust antenna tilt")
    if row['SINR'] < 10:
        actions.append("Optimize handover thresholds")
    if row['throughput_Mbps'] < 25 and row['call_drops'] > 0:
        actions.append("Investigate backhaul congestion")
    return ", ".join(actions) if actions else "No action needed"

