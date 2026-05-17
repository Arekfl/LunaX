import { useCallback, useEffect, useMemo, useState } from "react";
import L from "leaflet";
import { ImageOverlay, MapContainer, Rectangle, useMap } from "react-leaflet";

const IMAGE_BOUNDS = [
  [0, 0],
  [1024, 2048],
];

const IMAGE_HEIGHT = IMAGE_BOUNDS[1][0];
const IMAGE_WIDTH = IMAGE_BOUNDS[1][1];
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const DETECTION_STATUSES = ["confirmed", "to_verify", "rejected"];
const DEFAULT_DETECTION_STATUS = "to_verify";

const GRID_SIZE = 4;
const CELL_HEIGHT = IMAGE_HEIGHT / GRID_SIZE;
const CELL_WIDTH = IMAGE_WIDTH / GRID_SIZE;

function buildSegments() {
  const segments = [];

  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let col = 0; col < GRID_SIZE; col += 1) {
      const yMin = row * CELL_HEIGHT;
      const xMin = col * CELL_WIDTH;
      const yMax = yMin + CELL_HEIGHT;
      const xMax = xMin + CELL_WIDTH;

      segments.push({
        id: `segment-${row + 1}-${col + 1}`,
        row,
        col,
        bounds: [
          [yMin, xMin],
          [yMax, xMax],
        ],
      });
    }
  }

  return segments;
}

function boundsToCoords(bounds) {
  const [[yMin, xMin], [yMax, xMax]] = bounds;
  return { xMin, yMin, xMax, yMax };
}

function detectionToBounds(detection) {
  const { x, y, width, height } = detection.bbox;
  return [
    [y, x],
    [y + height, x + width],
  ];
}

function applyStatusesToDetections(detectionList, statusMap) {
  return detectionList.map((detection) => ({
    ...detection,
    status: statusMap[detection.detection_id] ?? detection.status ?? DEFAULT_DETECTION_STATUS,
  }));
}

function FitBoundsOnChange({ bounds }) {
  const map = useMap();

  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, { padding: [20, 20], animate: true });
    }
  }, [map, bounds]);

  return null;
}

function HomeControl({ onHomeClick }) {
  const map = useMap();

  useEffect(() => {
    const homeControl = L.control({ position: "topleft" });

    homeControl.onAdd = () => {
      const container = L.DomUtil.create("div", "leaflet-bar");
      const button = L.DomUtil.create("a", "leaflet-control-home", container);

      button.href = "#";
      button.title = "Pokaz cala mape";
      button.innerHTML = "&#8962;";

      L.DomEvent.disableClickPropagation(container);
      L.DomEvent.on(button, "click", (event) => {
        L.DomEvent.preventDefault(event);
        onHomeClick();
      });

      return container;
    };

    homeControl.addTo(map);

    return () => {
      homeControl.remove();
    };
  }, [map, onHomeClick]);

  return null;
}

export default function App() {
  const segments = useMemo(() => buildSegments(), []);
  const [detections, setDetections] = useState([]);
  const [isLoadingDetections, setIsLoadingDetections] = useState(false);
  const [storedStatuses, setStoredStatuses] = useState({});
  const [statusFilter, setStatusFilter] = useState("to_verify");
  const [hoveredSegmentId, setHoveredSegmentId] = useState(null);
  const [selectedSegment, setSelectedSegment] = useState(null);
  const [selectedDetection, setSelectedDetection] = useState(null);
  const [focusBounds, setFocusBounds] = useState(IMAGE_BOUNDS);
  const [chosenMessage, setChosenMessage] = useState("");
  const [manualCoords, setManualCoords] = useState({
    xMin: "0",
    yMin: "0",
    xMax: String(IMAGE_WIDTH),
    yMax: String(IMAGE_HEIGHT),
  });

  const selectedCoords = selectedSegment ? boundsToCoords(selectedSegment.bounds) : null;

  const filteredDetections = useMemo(
    () => detections.filter((detection) => detection.status === statusFilter),
    [detections, statusFilter]
  );

  useEffect(() => {
    const fetchStoredStatuses = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/detections/statuses`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const payload = await response.json();
        if (payload && typeof payload === "object" && !Array.isArray(payload)) {
          setStoredStatuses(payload);
        }
      } catch (error) {
        console.warn("Nie udalo sie pobrac zapisanych statusow detekcji:", error);
      }
    };

    fetchStoredStatuses();
  }, []);

  useEffect(() => {
    if (Object.keys(storedStatuses).length === 0) {
      return;
    }

    setDetections((prevDetections) => applyStatusesToDetections(prevDetections, storedStatuses));
  }, [storedStatuses]);

  useEffect(() => {
    if (!selectedDetection) {
      return;
    }

    const isSelectedVisible = filteredDetections.some(
      (detection) => detection.detection_id === selectedDetection.detection_id
    );

    if (!isSelectedVisible) {
      setSelectedDetection(null);
    }
  }, [filteredDetections, selectedDetection]);

  const handleResetHomeView = useCallback(() => {
    setSelectedSegment(null);
    setFocusBounds(IMAGE_BOUNDS);
    setManualCoords({
      xMin: "0",
      yMin: "0",
      xMax: String(IMAGE_WIDTH),
      yMax: String(IMAGE_HEIGHT),
    });
    setChosenMessage("Widok zresetowany do calej mapy.");
  }, []);

  const handleSelectSegment = (segment) => {
    setSelectedSegment(segment);
    setFocusBounds(segment.bounds);
    setChosenMessage("");

    const coords = boundsToCoords(segment.bounds);
    setManualCoords({
      xMin: String(coords.xMin),
      yMin: String(coords.yMin),
      xMax: String(coords.xMax),
      yMax: String(coords.yMax),
    });
  };

  const handleChooseArea = async () => {
    if (!selectedSegment) {
      setChosenMessage("Najpierw wybierz segment na mapie lub wpisz wspolrzedne.");
      return;
    }

    setIsLoadingDetections(true);

    try {
      const response = await fetch(`${API_BASE_URL}/analysis/run`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          region_id: selectedSegment.id,
          confidence_threshold: 0.5,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const payload = await response.json();
      const apiDetections = Array.isArray(payload.detections) ? payload.detections : [];
      const detectionsWithFallbackStatus = apiDetections.map((detection, index) => ({
        ...detection,
        status: DETECTION_STATUSES[index % DETECTION_STATUSES.length],
      }));
      const detectionsWithStatus = applyStatusesToDetections(
        detectionsWithFallbackStatus,
        storedStatuses
      );

      setDetections(detectionsWithStatus);
      setSelectedDetection(null);

      if (detectionsWithStatus.length === 0) {
        setChosenMessage("Analiza zakonczona. Brak detekcji dla wybranego obszaru.");
        return;
      }

      setChosenMessage(`Analiza zakonczona. Pobrano ${detectionsWithStatus.length} detekcji.`);
    } catch (error) {
      console.error("Blad podczas pobierania detekcji:", error);
      setChosenMessage("Nie udalo sie pobrac detekcji z backendu FastAPI.");
      setDetections([]);
      setSelectedDetection(null);
    } finally {
      setIsLoadingDetections(false);
    }
  };

  const handleGoToManual = () => {
    const xMin = Number(manualCoords.xMin);
    const yMin = Number(manualCoords.yMin);
    const xMax = Number(manualCoords.xMax);
    const yMax = Number(manualCoords.yMax);

    const hasInvalidValues =
      [xMin, yMin, xMax, yMax].some((value) => Number.isNaN(value)) ||
      xMin < 0 ||
      yMin < 0 ||
      xMax > IMAGE_WIDTH ||
      yMax > IMAGE_HEIGHT ||
      xMax <= xMin ||
      yMax <= yMin;

    if (hasInvalidValues) {
      setChosenMessage(
        `Niepoprawne wspolrzedne. Zakres: x 0-${IMAGE_WIDTH}, y 0-${IMAGE_HEIGHT} i xMax>xMin, yMax>yMin.`
      );
      return;
    }

    const manualBounds = [
      [yMin, xMin],
      [yMax, xMax],
    ];

    setSelectedSegment({
      id: "manual",
      row: null,
      col: null,
      bounds: manualBounds,
    });
    setFocusBounds(manualBounds);
    setChosenMessage("Przejscie do recznie wskazanego obszaru.");
  };

  return (
    <div className="container-fluid py-3">
      <div className="row g-3">
        <div className="col-lg-9">
          <div className="map-shell border rounded shadow-sm">
            <MapContainer crs={L.CRS.Simple} bounds={IMAGE_BOUNDS} minZoom={-2} maxZoom={4}>
              <ImageOverlay url="/luna_0.jpg" bounds={IMAGE_BOUNDS} />
              <FitBoundsOnChange bounds={focusBounds} />
              <HomeControl onHomeClick={handleResetHomeView} />

              {segments.map((segment) => {
                const isHovered = hoveredSegmentId === segment.id;
                const isSelected = selectedSegment?.id === segment.id;

                let color = "#0d6efd";
                if (isSelected) {
                  color = "#dc3545";
                } else if (isHovered) {
                  color = "#fd7e14";
                }

                return (
                  <Rectangle
                    key={segment.id}
                    bounds={segment.bounds}
                    pathOptions={{
                      color,
                      weight: isSelected ? 3 : 1,
                      fillColor: color,
                      fillOpacity: isSelected ? 0.22 : isHovered ? 0.16 : 0.06,
                    }}
                    eventHandlers={{
                      mouseover: () => setHoveredSegmentId(segment.id),
                      mouseout: () => setHoveredSegmentId(null),
                      click: () => handleSelectSegment(segment),
                    }}
                  />
                );
              })}

              {filteredDetections.map((detection) => {
                const isSelected = selectedDetection?.detection_id === detection.detection_id;

                return (
                  <Rectangle
                    key={detection.detection_id}
                    bounds={detectionToBounds(detection)}
                    pathOptions={{
                      color: isSelected ? "#ffc107" : "#20c997",
                      weight: isSelected ? 3 : 1,
                      fillColor: isSelected ? "#ffc107" : "#20c997",
                      fillOpacity: isSelected ? 0.28 : 0.08,
                      dashArray: isSelected ? null : "4 4",
                    }}
                    eventHandlers={{
                      click: () => setSelectedDetection(detection),
                    }}
                  />
                );
              })}
            </MapContainer>
          </div>
        </div>

        <div className="col-lg-3">
          <div className="card shadow-sm">
            <div className="card-body">
              <h5 className="card-title">Panel obszaru</h5>

              <div className="small text-muted mb-2">Wybrany segment</div>
              {selectedSegment ? (
                <div className="mb-3">
                  <div><strong>ID:</strong> {selectedSegment.id}</div>
                  {selectedCoords && (
                    <div className="mt-2">
                      <div>xMin: {selectedCoords.xMin}</div>
                      <div>yMin: {selectedCoords.yMin}</div>
                      <div>xMax: {selectedCoords.xMax}</div>
                      <div>yMax: {selectedCoords.yMax}</div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="mb-3 text-muted">Brak wybranego segmentu.</div>
              )}

              <button className="btn btn-primary w-100 mb-3" onClick={handleChooseArea}>
                {isLoadingDetections ? "Analizowanie..." : "Wybierz obszar"}
              </button>

              <div className="small text-muted mb-2">Przejdz do wspolrzednych</div>
              <div className="row g-2">
                <div className="col-6">
                  <input
                    className="form-control form-control-sm"
                    type="number"
                    step="any"
                    value={manualCoords.xMin}
                    onChange={(event) =>
                      setManualCoords((prev) => ({ ...prev, xMin: event.target.value }))
                    }
                    placeholder="xMin"
                  />
                </div>
                <div className="col-6">
                  <input
                    className="form-control form-control-sm"
                    type="number"
                    step="any"
                    value={manualCoords.yMin}
                    onChange={(event) =>
                      setManualCoords((prev) => ({ ...prev, yMin: event.target.value }))
                    }
                    placeholder="yMin"
                  />
                </div>
                <div className="col-6">
                  <input
                    className="form-control form-control-sm"
                    type="number"
                    step="any"
                    value={manualCoords.xMax}
                    onChange={(event) =>
                      setManualCoords((prev) => ({ ...prev, xMax: event.target.value }))
                    }
                    placeholder="xMax"
                  />
                </div>
                <div className="col-6">
                  <input
                    className="form-control form-control-sm"
                    type="number"
                    step="any"
                    value={manualCoords.yMax}
                    onChange={(event) =>
                      setManualCoords((prev) => ({ ...prev, yMax: event.target.value }))
                    }
                    placeholder="yMax"
                  />
                </div>
              </div>

              <button className="btn btn-outline-secondary w-100 mt-3" onClick={handleGoToManual}>
                Przejdz
              </button>

              {chosenMessage && <div className="alert alert-info py-2 mt-3 mb-0">{chosenMessage}</div>}

              <hr className="my-4" />
              <h6 className="mb-3">Detekcje</h6>

              <div className="btn-group btn-group-sm w-100 mb-3" role="group" aria-label="Filtr statusu detekcji">
                <button
                  type="button"
                  className={`btn ${statusFilter === "confirmed" ? "btn-success" : "btn-outline-success"}`}
                  onClick={() => setStatusFilter("confirmed")}
                >
                  confirmed
                </button>
                <button
                  type="button"
                  className={`btn ${statusFilter === "to_verify" ? "btn-warning" : "btn-outline-warning"}`}
                  onClick={() => setStatusFilter("to_verify")}
                >
                  to_verify
                </button>
                <button
                  type="button"
                  className={`btn ${statusFilter === "rejected" ? "btn-danger" : "btn-outline-danger"}`}
                  onClick={() => setStatusFilter("rejected")}
                >
                  rejected
                </button>
              </div>

              {detections.length === 0 ? (
                <div className="small text-muted">Brak detekcji. Kliknij "Wybierz obszar".</div>
              ) : filteredDetections.length === 0 ? (
                <div className="small text-muted">Brak detekcji dla statusu: {statusFilter}.</div>
              ) : (
                <div className="list-group">
                  {filteredDetections.map((detection) => {
                    const isSelected =
                      selectedDetection?.detection_id === detection.detection_id;

                    return (
                      <button
                        key={detection.detection_id}
                        type="button"
                        onClick={() => setSelectedDetection(detection)}
                        className={`list-group-item list-group-item-action text-start ${
                          isSelected ? "bg-primary-subtle border-primary" : ""
                        }`}
                      >
                        <div><strong>{detection.detection_id}</strong></div>
                        <div className="small">status: {detection.status}</div>
                        <div className="small text-muted">
                          confidence: {Number(detection.confidence).toFixed(2)}
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
