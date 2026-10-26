# AI Route Finder System

A browser-based demonstration of route optimization using Dijkstra's algorithm. Users can choose a start point, destination, time-of-day traffic scenario, and route priority (Speed or Safety). The app visualizes the selected path and displays distance, estimated travel time, and average road-safety score.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Uses a sample city graph with eight locations and twelve roads.
- Traffic is simulated using fixed sample values and a rush-hour adjustment.
- It does not use live GPS, live traffic, or a real-world map API.
- The original desktop implementation is written in `ai_route_finder.py`; `app.py` is the Streamlit web interface for deployment.
