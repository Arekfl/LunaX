import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import { MapContainer, Rectangle, useMap, useMapEvents, WMSTileLayer } from "react-leaflet";

const GEO_BOUNDS = [
  [-90, -180],
  [90, 180],
];

const LON_MIN = GEO_BOUNDS[0][1];
const LON_MAX = GEO_BOUNDS[1][1];
const LAT_MIN = GEO_BOUNDS[0][0];
const LAT_MAX = GEO_BOUNDS[1][0];
const GEO_WIDTH = LON_MAX - LON_MIN;
const GEO_HEIGHT = LAT_MAX - LAT_MIN;
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const DEFAULT_DETECTION_STATUS = "to_verify";
const STATUS_COLOR_MAP = {
  confirmed: "#198754",
  to_verify: "#ffc107",
  rejected: "#dc3545",
};
const STATUS_BADGE_CLASS_MAP = {
  confirmed: "text-bg-success",
  to_verify: "text-bg-warning",
  rejected: "text-bg-danger",
};
const DETECTION_BBOX_PROXIMITY_THRESHOLD = 12;
const NO_DETECTIONS_FILTER = "no_detections";
const ALL_ANALYSIS_IMAGES_FILTER = "all";
const RESOLUTION_DESCRIPTION_MAP = {
  preview: "Szybki podglad, nizsza dokladnosc.",
  detail: "Zbalansowany tryb do codziennej analizy.",
  ultra: "Najwyzsza dokladnosc, najdluzszy czas analizy.",
};
const RESOLUTION_MPP_MAP = {
  preview: 59.22,
  detail: 15.79,
  ultra: 0.87,
};
const WMS_SOURCE_OPTIONS = [
  { value: "usgs", label: "USGS" },
  { value: "lroc_ildi", label: "LROC IM-LDI" },
];
const WMS_LAYER_OPTIONS_BY_SOURCE = {
  usgs: [
    { value: "auto", label: "auto" },
    { value: "LROC_WAC", label: "LROC_WAC" },
    { value: "KaguyaTC_Ortho", label: "KaguyaTC_Ortho" },
  ],
  lroc_ildi: [
    { value: "auto", label: "auto" },
    { value: "luna_wac_global", label: "luna_wac_global" },
    { value: "luna_nac_gigapan", label: "luna_nac_gigapan" },
    { value: "luna_pds_nac_stamp", label: "luna_pds_nac_stamp" },
  ],
};

const GRID_SIZE = 4;
const GRID_ROWS = GRID_SIZE;
const GRID_COLS = GRID_SIZE;

function buildGridCells(bounds, level) {
  const rows = GRID_ROWS;
  const cols = GRID_COLS;
  const [[latMin, lonMin], [latMax, lonMax]] = bounds;
  const cellHeight = (latMax - latMin) / rows;
  const cellWidth = (lonMax - lonMin) / cols;
  const segments = [];

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const yMin = latMin + row * cellHeight;
      const xMin = lonMin + col * cellWidth;
      const yMax = row === rows - 1 ? latMax : latMin + (row + 1) * cellHeight;
      const xMax = col === cols - 1 ? lonMax : lonMin + (col + 1) * cellWidth;

      segments.push({
        id: `level-${level}-segment-${row + 1}-${col + 1}`,
        level,
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

function parseBBoxToMinMax(bbox) {
  if (Array.isArray(bbox) && bbox.length === 4) {
    const [xMin, yMin, xMax, yMax] = bbox;
    return { xMin, yMin, xMax, yMax };
  }

  if (!bbox || typeof bbox !== "object") {
    return null;
  }

  if (
    typeof bbox.xMin === "number" &&
    typeof bbox.yMin === "number" &&
    typeof bbox.xMax === "number" &&
    typeof bbox.yMax === "number"
  ) {
    return {
      xMin: bbox.xMin,
      yMin: bbox.yMin,
      xMax: bbox.xMax,
      yMax: bbox.yMax,
    };
  }

  if (
    typeof bbox.x === "number" &&
    typeof bbox.y === "number" &&
    typeof bbox.width === "number" &&
    typeof bbox.height === "number"
  ) {
    return {
      xMin: bbox.x,
      yMin: bbox.y,
      xMax: bbox.x + bbox.width,
      yMax: bbox.y + bbox.height,
    };
  }

  return null;
}

function isBoundsInsideImage(bounds) {
  if (!bounds) {
    return false;
  }

  const [[yMin, xMin], [yMax, xMax]] = bounds;

  return (
    Number.isFinite(yMin) &&
    Number.isFinite(xMin) &&
    Number.isFinite(yMax) &&
    Number.isFinite(xMax) &&
    yMin >= GEO_BOUNDS[0][0] &&
    xMin >= GEO_BOUNDS[0][1] &&
    yMax <= GEO_BOUNDS[1][0] &&
    xMax <= GEO_BOUNDS[1][1] &&
    yMax > yMin &&
    xMax > xMin
  );
}

function detectionToBounds(detection) {
  const bboxMinMax = parseBBoxToMinMax(detection?.bbox);
  if (!bboxMinMax) {
    return null;
  }

  const { xMin, yMin, xMax, yMax } = bboxMinMax;
  const bounds = [
    [yMin, xMin],
    [yMax, xMax],
  ];

  return isBoundsInsideImage(bounds) ? bounds : null;
}

function areBBoxesClose(leftBBox, rightBBox, threshold) {
  const left = parseBBoxToMinMax(leftBBox);
  const right = parseBBoxToMinMax(rightBBox);

  if (!left || !right) {
    return false;
  }

  return (
    Math.abs(left.xMin - right.xMin) < threshold &&
    Math.abs(left.yMin - right.yMin) < threshold &&
    Math.abs(left.xMax - right.xMax) < threshold &&
    Math.abs(left.yMax - right.yMax) < threshold
  );
}

function deduplicateDetectionsByProximity(detectionList, threshold) {
  const deduplicated = [];

  for (const detection of detectionList) {
    const matchIndex = deduplicated.findIndex((existing) => {
      if (existing.class !== detection.class) {
        return false;
      }

      return areBBoxesClose(existing.bbox, detection.bbox, threshold);
    });

    if (matchIndex === -1) {
      deduplicated.push(detection);
      continue;
    }

    const existing = deduplicated[matchIndex];
    if (Number(detection.confidence) > Number(existing.confidence)) {
      deduplicated[matchIndex] = detection;
    }
  }

  return deduplicated;
}

function getDisplayDetectionsForStatus(detectionList, status) {
  return deduplicateDetectionsByProximity(
    detectionList
      .filter((detection) => detectionToBounds(detection))
      .filter((detection) => detection.status === status),
    DETECTION_BBOX_PROXIMITY_THRESHOLD
  );
}

function isNoCoverageErrorMessage(message) {
  if (typeof message !== "string") {
    return false;
  }

  const normalizedMessage = message.toLowerCase();
  return (
    normalizedMessage.includes("no valid imagery") ||
    normalizedMessage.includes("fully transparent") ||
    normalizedMessage.includes("flat (single-value image)")
  );
}

function getDetectionUniqueId(detection) {
  const { analysis_id: analysisId, detection_id: detectionId, bbox } = detection;
  return [analysisId ?? "no-analysis", detectionId, bbox.x, bbox.y, bbox.width, bbox.height].join("|");
}

function isSameDetection(leftDetection, rightDetection) {
  if (!leftDetection || !rightDetection) {
    return false;
  }

  return getDetectionUniqueId(leftDetection) === getDetectionUniqueId(rightDetection);
}

function resolveDetectionStatus(detection, statusMap) {
  return statusMap[detection.detection_id] ?? detection.status ?? DEFAULT_DETECTION_STATUS;
}

function applyStatusesToDetections(detectionList, statusMap) {
  return detectionList.map((detection) => ({
    ...detection,
    status: statusMap[detection.detection_id] ?? detection.status ?? DEFAULT_DETECTION_STATUS,
  }));
}

function getStatusColor(status) {
  return STATUS_COLOR_MAP[status] ?? STATUS_COLOR_MAP[DEFAULT_DETECTION_STATUS];
}

function getStatusBadgeClass(status) {
  return STATUS_BADGE_CLASS_MAP[status] ?? "text-bg-secondary";
}

function getFileNameFromPath(filePath) {
  if (!filePath) {
    return "";
  }

  const normalizedPath = String(filePath).replaceAll("\\", "/");
  const parts = normalizedPath.split("/").filter(Boolean);
  return parts[parts.length - 1] ?? "";
}

function getAnalysisImageUrl(imageId) {
  if (!imageId) {
    return null;
  }

  return `${API_BASE_URL}/analysis-images/image/${encodeURIComponent(imageId)}`;
}

function getDetectionPreviewUrl(detection) {
  const bboxMinMax = parseBBoxToMinMax(detection?.bbox);
  if (!bboxMinMax) {
    return null;
  }

  const { xMin, yMin, xMax, yMax } = bboxMinMax;
  const params = new URLSearchParams({
    map: "/maps/earth/moon_simp_cyl.map",
    SERVICE: "WMS",
    VERSION: "1.1.1",
    REQUEST: "GetMap",
    LAYERS: "KaguyaTC_Ortho",
    STYLES: "",
    FORMAT: "image/png",
    SRS: "EPSG:4326",
    BBOX: `${xMin},${yMin},${xMax},${yMax}`,
    WIDTH: "512",
    HEIGHT: "512",
  });

  return `https://planetarymaps.usgs.gov/cgi-bin/mapserv?${params.toString()}`;
}

function FitBoundsOnChange({ bounds }) {
  const map = useMap();

  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, { padding: [0, 0], animate: false });
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

function ZoomOutLevelControl({ currentLevel, onStepOut, suppressZoomOutRef }) {
  const lastZoomRef = useRef(null);
  const map = useMapEvents({
    zoomend() {
      const currentZoom = map.getZoom();
      const previousZoom = lastZoomRef.current;

      if (suppressZoomOutRef.current > 0) {
        suppressZoomOutRef.current -= 1;
        lastZoomRef.current = currentZoom;
        return;
      }

      if (typeof previousZoom === "number" && currentZoom < previousZoom && currentLevel > 0) {
        onStepOut();
      }

      lastZoomRef.current = currentZoom;
    },
  });

  useEffect(() => {
    lastZoomRef.current = map.getZoom();
  }, [map]);

  return null;
}

export default function App() {
  const mapRef = useRef(null);
  const detectionListRef = useRef(null);
  const detectionItemRefs = useRef(new Map());
  const suppressZoomOutRef = useRef(0);
  const [currentLevel, setCurrentLevel] = useState(0);
  const [isLevelLocked, setIsLevelLocked] = useState(false);
  const [selectedBBox, setSelectedBBox] = useState(GEO_BOUNDS);
  const [bboxHistory, setBBoxHistory] = useState([GEO_BOUNDS]);
  const [gridCells, setGridCells] = useState(() => buildGridCells(GEO_BOUNDS, 0));
  const [detections, setDetections] = useState([]);
  const [analysisImages, setAnalysisImages] = useState([]);
  const [analysisImagesFilter, setAnalysisImagesFilter] = useState(NO_DETECTIONS_FILTER);
  const [selectedNoDetectionImage, setSelectedNoDetectionImage] = useState(null);
  const [currentAnalysisId, setCurrentAnalysisId] = useState(null);
  const [isLoadingDetections, setIsLoadingDetections] = useState(false);
  const [analysisOverlayBounds, setAnalysisOverlayBounds] = useState(null);
  const [analysisStatus, setAnalysisStatus] = useState(null);
  const [showBboxes, setShowBboxes] = useState(true);
  const [viewMode, setViewMode] = useState("map");
  const [resolutionMode, setResolutionMode] = useState("detail");
  const [wmsSource, setWmsSource] = useState("usgs");
  const [wmsLayer, setWmsLayer] = useState("auto");
  const [numSamples, setNumSamples] = useState(5);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [storedStatuses, setStoredStatuses] = useState({});
  const [statusFilter, setStatusFilter] = useState("to_verify");
  const [hoveredSegmentId, setHoveredSegmentId] = useState(null);
  const [selectedSegment, setSelectedSegment] = useState(null);
  const [selectedDetection, setSelectedDetection] = useState(null);
  const [hoveredDetectionId, setHoveredDetectionId] = useState(null);
  const [inputComment, setInputComment] = useState("");
  const [editingDetectionId, setEditingDetectionId] = useState(null);
  const [focusBounds, setFocusBounds] = useState(GEO_BOUNDS);
  const [chosenMessage, setChosenMessage] = useState("");
  const [manualCoords, setManualCoords] = useState({
    xMin: String(LON_MIN),
    yMin: String(LAT_MIN),
    xMax: String(LON_MAX),
    yMax: String(LAT_MAX),
  });

  const selectedCoords = selectedSegment ? boundsToCoords(selectedSegment.bounds) : null;
  const isAnalysisLoading = isLoadingDetections || analysisStatus === "loading";
  const isNoDetectionsFilterSelected = statusFilter === NO_DETECTIONS_FILTER;
  const isNoCoverageChosenMessage =
    typeof chosenMessage === "string" &&
    chosenMessage.startsWith("Brak pokrycia danych dla tej warstwy i obszaru.");
  const selectedWmsLayerOptions = WMS_LAYER_OPTIONS_BY_SOURCE[wmsSource] ?? [];
  const isGlobalMosaicLayerSelected =
    (wmsSource === "usgs" && wmsLayer === "LROC_WAC") ||
    (wmsSource === "lroc_ildi" && wmsLayer === "luna_wac_global");

  useEffect(() => {
    setGridCells(buildGridCells(selectedBBox, currentLevel));
    setHoveredSegmentId(null);
  }, [selectedBBox, currentLevel]);

  useEffect(() => {
    const isLayerValidForSource = selectedWmsLayerOptions.some((option) => option.value === wmsLayer);
    if (!isLayerValidForSource) {
      setWmsLayer("auto");
    }
  }, [selectedWmsLayerOptions, wmsLayer]);

  const filteredDetections = useMemo(() => {
    const statusFilteredDetections = detections
      .map((detection) => ({
        ...detection,
        status: resolveDetectionStatus(detection, storedStatuses),
      }));

    return getDisplayDetectionsForStatus(statusFilteredDetections, statusFilter);
  }, [detections, statusFilter, storedStatuses]);

  const visibleAnalysisImages = useMemo(() => {
    if (analysisImagesFilter === ALL_ANALYSIS_IMAGES_FILTER) {
      return analysisImages;
    }

    return analysisImages.filter(
      (image) => (typeof image.status === "string" ? image.status : NO_DETECTIONS_FILTER) === NO_DETECTIONS_FILTER
    );
  }, [analysisImages, analysisImagesFilter]);

  const groupedAnalysisImages = useMemo(() => {
    const groups = new Map();

    for (const image of visibleAnalysisImages) {
      const analysisId =
        typeof image.analysis_id === "string" && image.analysis_id.trim().length > 0
          ? image.analysis_id
          : "legacy_no_analysis_id";

      if (!groups.has(analysisId)) {
        groups.set(analysisId, []);
      }

      groups.get(analysisId).push(image);
    }

    return Array.from(groups.entries()).map(([analysisId, images]) => ({
      analysisId,
      images,
    }));
  }, [visibleAnalysisImages]);

  const detectionSectionCount = isNoDetectionsFilterSelected
    ? visibleAnalysisImages.length
    : filteredDetections.length;

  const fetchDetectionStatuses = useCallback(async () => {
    const statusesResponse = await fetch(`${API_BASE_URL}/detections/statuses`);

    if (!statusesResponse.ok) {
      throw new Error(`Detection statuses HTTP ${statusesResponse.status}`);
    }

    const statusesPayload = await statusesResponse.json();
    const statusMap =
      statusesPayload && typeof statusesPayload === "object" && !Array.isArray(statusesPayload)
        ? statusesPayload
        : {};

    setStoredStatuses(statusMap);
    return statusMap;
  }, []);

  const fetchDetectionsAndStatuses = useCallback(async () => {
    const [detectionsResponse, statusMap] = await Promise.all([
      fetch(`${API_BASE_URL}/detections/query`),
      fetchDetectionStatuses(),
    ]);

    if (!detectionsResponse.ok) {
      throw new Error(`Detections query HTTP ${detectionsResponse.status}`);
    }

    const detectionsPayload = await detectionsResponse.json();

    const queriedDetections = Array.isArray(detectionsPayload) ? detectionsPayload : [];

    const mergedDetections = applyStatusesToDetections(queriedDetections, statusMap);
    setDetections(mergedDetections);

    return mergedDetections;
  }, [fetchDetectionStatuses]);

  const fetchAnalysisImages = useCallback(async () => {
    const response = await fetch(`${API_BASE_URL}/analysis-images/query`);

    if (!response.ok) {
      throw new Error(`Analysis-images query HTTP ${response.status}`);
    }

    const payload = await response.json();
    const images = Array.isArray(payload) ? payload : [];
    setAnalysisImages(images);
    return images;
  }, []);

  useEffect(() => {
    const loadInitialData = async () => {
      try {
        await Promise.all([fetchDetectionsAndStatuses(), fetchAnalysisImages()]);
      } catch (error) {
        console.warn("Nie udalo sie pobrac danych poczatkowych:", error);
      }
    };

    loadInitialData();
  }, [fetchDetectionsAndStatuses, fetchAnalysisImages]);

  useEffect(() => {
    if (!selectedDetection) {
      return;
    }

    const isSelectedVisible = filteredDetections.some(
      (detection) => isSameDetection(detection, selectedDetection)
    );

    if (!isSelectedVisible) {
      setSelectedDetection(null);
    }
  }, [filteredDetections, selectedDetection]);

  useEffect(() => {
    if (!selectedDetection) {
      return;
    }

    const selectedIndex = filteredDetections.findIndex((detection) =>
      isSameDetection(detection, selectedDetection)
    );
    if (selectedIndex < 0) {
      return;
    }

    const selectedRenderKey = `${getDetectionUniqueId(filteredDetections[selectedIndex])}|${selectedIndex}`;
    const selectedItem = detectionItemRefs.current.get(selectedRenderKey);
    const container = detectionListRef.current;
    if (!selectedItem || !container) {
      return;
    }

    const containerRect = container.getBoundingClientRect();
    const itemRect = selectedItem.getBoundingClientRect();
    const padding = 8;

    if (itemRect.top < containerRect.top) {
      container.scrollTop -= containerRect.top - itemRect.top + padding;
      return;
    }

    if (itemRect.bottom > containerRect.bottom) {
      container.scrollTop += itemRect.bottom - containerRect.bottom + padding;
    }
  }, [filteredDetections, selectedDetection]);

  useEffect(() => {
    if (!hoveredDetectionId) {
      return;
    }

    const isHoveredVisible = filteredDetections.some(
      (detection) => getDetectionUniqueId(detection) === hoveredDetectionId
    );

    if (!isHoveredVisible) {
      setHoveredDetectionId(null);
    }
  }, [filteredDetections, hoveredDetectionId]);

  useEffect(() => {
    if (!editingDetectionId) {
      return;
    }

    const isEditingVisible = filteredDetections.some(
      (detection) => detection.detection_id === editingDetectionId
    );

    if (!isEditingVisible) {
      setEditingDetectionId(null);
      setInputComment("");
    }
  }, [filteredDetections, editingDetectionId]);

  useEffect(() => {
    if (!selectedNoDetectionImage) {
      return;
    }

    const stillExists = visibleAnalysisImages.some(
      (image) => image.image_id === selectedNoDetectionImage.image_id
    );

    if (!stillExists) {
      setSelectedNoDetectionImage(null);
    }
  }, [visibleAnalysisImages, selectedNoDetectionImage]);

  useEffect(() => {
    if (!isNoDetectionsFilterSelected && selectedNoDetectionImage) {
      setSelectedNoDetectionImage(null);
    }
  }, [isNoDetectionsFilterSelected, selectedNoDetectionImage]);

  const handleResetHomeView = useCallback(() => {
    setCurrentLevel(0);
    setSelectedBBox(GEO_BOUNDS);
    setBBoxHistory([GEO_BOUNDS]);
    setGridCells(buildGridCells(GEO_BOUNDS, 0));
    setSelectedSegment(null);
    setFocusBounds(GEO_BOUNDS);
    setManualCoords({
      xMin: String(LON_MIN),
      yMin: String(LAT_MIN),
      xMax: String(LON_MAX),
      yMax: String(LAT_MAX),
    });
    setChosenMessage("Widok zresetowany do calej mapy.");
  }, []);

  const handleStepOutLevel = useCallback(() => {
    setBBoxHistory((prevHistory) => {
      if (prevHistory.length <= 1) {
        return prevHistory;
      }

      const nextHistory = prevHistory.slice(0, -1);
      const parentBounds = nextHistory[nextHistory.length - 1];

      // Ignore only the immediate zoom event from this programmatic fitBounds.
      suppressZoomOutRef.current = 1;

      setCurrentLevel(nextHistory.length - 1);
      setSelectedBBox(parentBounds);
      setFocusBounds(parentBounds);
      setSelectedSegment(null);
      setSelectedDetection(null);
      setHoveredDetectionId(null);

      const coords = boundsToCoords(parentBounds);
      setManualCoords({
        xMin: String(coords.xMin),
        yMin: String(coords.yMin),
        xMax: String(coords.xMax),
        yMax: String(coords.yMax),
      });

      return nextHistory;
    });
  }, []);

  const handleSelectSegment = (segment) => {
    setSelectedSegment(segment);
    setChosenMessage("");

    if (!isLevelLocked) {
      setCurrentLevel((prevLevel) => prevLevel + 1);
      setSelectedBBox(segment.bounds);
      setBBoxHistory((prevHistory) => [...prevHistory, segment.bounds]);
      setFocusBounds(segment.bounds);
    }

    const coords = boundsToCoords(segment.bounds);
    setManualCoords({
      xMin: String(coords.xMin),
      yMin: String(coords.yMin),
      xMax: String(coords.xMax),
      yMax: String(coords.yMax),
    });
  };

  const handleSelectDetection = useCallback((detection) => {
    const bboxBounds = detectionToBounds(detection);
    if (!bboxBounds) {
      return;
    }

    setSelectedDetection(detection);

    const mapInstance = mapRef.current;
    if (!mapInstance) {
      return;
    }

    mapInstance.fitBounds(bboxBounds, { padding: [20, 20], animate: true });
  }, []);

  const handleChooseArea = async () => {
    if (!selectedSegment) {
      setChosenMessage("Najpierw wybierz segment na mapie lub wpisz wspolrzedne.");
      setAnalysisStatus("error");
      return;
    }

    setIsLoadingDetections(true);
    setAnalysisStatus("loading");
    setAnalysisOverlayBounds(selectedSegment.bounds);

    const [[yMin, xMin], [yMax, xMax]] = selectedSegment.bounds;
    const analysisBbox = [xMin, yMin, xMax, yMax];

    try {
      const response = await fetch(`${API_BASE_URL}/analysis/run`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          resolutionMode,
          wmsSource,
          wmsLayer,
          numSamples,
          confidenceThreshold,
          bbox: analysisBbox,
        }),
      });

      if (!response.ok) {
        let errorDetail = `HTTP ${response.status}`;
        try {
          const errorPayload = await response.json();
          if (errorPayload && typeof errorPayload.detail === "string") {
            errorDetail = `${errorDetail}: ${errorPayload.detail}`;
          }
        } catch {
          // Ignore parse failures and keep default HTTP message.
        }
        throw new Error(errorDetail);
      }

      const runPayload = await response.json();
      const analysisId =
        runPayload && typeof runPayload === "object" && typeof runPayload.analysis_id === "string"
          ? runPayload.analysis_id
          : null;
      const runDetections =
        runPayload && typeof runPayload === "object" && Array.isArray(runPayload.detections)
          ? runPayload.detections
          : [];

      if (!analysisId) {
        throw new Error("Missing analysis_id in /analysis/run response");
      }

      const statusMap = await fetchDetectionStatuses();
      const detectionsWithStatus = applyStatusesToDetections(runDetections, statusMap);
      const visibleWithCurrentFilter = getDisplayDetectionsForStatus(
        detectionsWithStatus,
        statusFilter
      );

      if (detectionsWithStatus.length > 0 && visibleWithCurrentFilter.length === 0) {
        const fallbackStatus = ["to_verify", "confirmed", "rejected"].find(
          (candidateStatus) =>
            getDisplayDetectionsForStatus(detectionsWithStatus, candidateStatus).length > 0
        );

        if (fallbackStatus) {
          setStatusFilter(fallbackStatus);
        }
      }

      setCurrentAnalysisId(analysisId);
      setDetections(detectionsWithStatus);
      try {
        await fetchAnalysisImages();
      } catch (refreshError) {
        console.warn("Nie udalo sie odswiezyc listy zapisanych obrazow analizy:", refreshError);
      }
      setSelectedDetection(null);
      setAnalysisStatus("success");

      if (detectionsWithStatus.length === 0) {
        setChosenMessage(`Analiza ${analysisId} zakonczona. Brak detekcji dla wybranego obszaru.`);
        return;
      }

      setChosenMessage(`Analiza ${analysisId} zakonczona. Pobrano ${detectionsWithStatus.length} detekcji.`);
    } catch (error) {
      console.error("Blad podczas pobierania detekcji:", error);
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Nie udalo sie pobrac detekcji z backendu FastAPI.";
      if (isNoCoverageErrorMessage(errorMessage)) {
        setChosenMessage(
          "Brak pokrycia danych dla tej warstwy i obszaru. Zmien warstwe, zrodlo WMS lub zaznacz inny obszar."
        );
      } else {
        setChosenMessage(`Nie udalo sie pobrac detekcji. Szczegoly: ${errorMessage}`);
      }
      setAnalysisStatus("error");
      setDetections([]);
      setSelectedDetection(null);
    } finally {
      setIsLoadingDetections(false);
      setAnalysisOverlayBounds(null);
    }
  };

  const handleLocalAnalysis = async () => {
    setIsLoadingDetections(true);
    setAnalysisStatus("loading");
    setAnalysisOverlayBounds(null);

    try {
      const response = await fetch(`${API_BASE_URL}/analysis/local-run`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          confidenceThreshold,
        }),
      });

      if (!response.ok) {
        let errorDetail = `HTTP ${response.status}`;
        try {
          const errorPayload = await response.json();
          if (errorPayload && typeof errorPayload.detail === "string") {
            errorDetail = `${errorDetail}: ${errorPayload.detail}`;
          }
        } catch {
          // Ignore parse failures and keep default HTTP message.
        }
        throw new Error(errorDetail);
      }

      const runPayload = await response.json();
      const analysisId =
        runPayload && typeof runPayload === "object" && typeof runPayload.analysis_id === "string"
          ? runPayload.analysis_id
          : null;
      const runDetections =
        runPayload && typeof runPayload === "object" && Array.isArray(runPayload.detections)
          ? runPayload.detections
          : [];

      if (!analysisId) {
        throw new Error("Missing analysis_id in /analysis/local-run response");
      }

      const statusMap = await fetchDetectionStatuses();
      const detectionsWithStatus = applyStatusesToDetections(runDetections, statusMap);
      const visibleWithCurrentFilter = getDisplayDetectionsForStatus(
        detectionsWithStatus,
        statusFilter
      );

      if (detectionsWithStatus.length > 0 && visibleWithCurrentFilter.length === 0) {
        const fallbackStatus = ["to_verify", "confirmed", "rejected"].find(
          (candidateStatus) =>
            getDisplayDetectionsForStatus(detectionsWithStatus, candidateStatus).length > 0
        );

        if (fallbackStatus) {
          setStatusFilter(fallbackStatus);
        }
      }

      setCurrentAnalysisId(analysisId);
      setDetections(detectionsWithStatus);
      setSelectedDetection(null);
      setAnalysisStatus("success");

      if (detectionsWithStatus.length === 0) {
        setChosenMessage(
          `Analiza lokalna ${analysisId} zakonczona. Brak detekcji w folderze validation.`
        );
        return;
      }

      setChosenMessage(
        `Analiza lokalna ${analysisId} zakonczona. Pobrano ${detectionsWithStatus.length} detekcji.`
      );
    } catch (error) {
      console.error("Blad podczas analizy lokalnej:", error);
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Nie udalo sie uruchomic analizy lokalnej z folderu validation.";
      setChosenMessage(`Nie udalo sie uruchomic analizy lokalnej. Szczegoly: ${errorMessage}`);
      setAnalysisStatus("error");
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
      xMin < LON_MIN ||
      yMin < LAT_MIN ||
      xMax > LON_MAX ||
      yMax > LAT_MAX ||
      xMax <= xMin ||
      yMax <= yMin;

    if (hasInvalidValues) {
      setChosenMessage(
        `Niepoprawne wspolrzedne. Zakres: lon ${LON_MIN} do ${LON_MAX}, lat ${LAT_MIN} do ${LAT_MAX}, oraz xMax>xMin i yMax>yMin.`
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

  const handleStartEditComment = (detection) => {
    setSelectedDetection(detection);
    setEditingDetectionId(detection.detection_id);
    setInputComment(detection.comment ?? "");
  };

  const handleSaveComment = async (detectionId) => {
    const targetDetectionId = editingDetectionId ?? detectionId;
    const currentComment = inputComment;

    if (!editingDetectionId && currentComment.trim().length === 0) {
      setChosenMessage("Wpisz komentarz przed zapisem.");
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/detections/${targetDetectionId}/comment`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ comment: currentComment }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const payload = await response.json();

      setDetections((prev) =>
        prev.map((detection) =>
          detection.detection_id === targetDetectionId
            ? { ...detection, comment: payload.comment }
            : detection
        )
      );

      setInputComment("");
      setEditingDetectionId(null);
      setChosenMessage(editingDetectionId ? "Komentarz zaktualizowany." : "Komentarz zapisany.");
    } catch (error) {
      console.error("Blad podczas zapisu komentarza:", error);
      setChosenMessage("Nie udalo sie zapisac komentarza.");
    }
  };

  const handleDeleteComment = async (detectionId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/detections/${detectionId}/comment`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ comment: "" }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      setDetections((prev) =>
        prev.map((detection) =>
          detection.detection_id === detectionId
            ? { ...detection, comment: "" }
            : detection
        )
      );

      if (editingDetectionId === detectionId) {
        setEditingDetectionId(null);
        setInputComment("");
      }

      setChosenMessage("Komentarz usuniety.");
    } catch (error) {
      console.error("Blad podczas usuwania komentarza:", error);
      setChosenMessage("Nie udalo sie usunac komentarza.");
    }
  };

  const handleOpenAnalysisImage = useCallback((image) => {
    setSelectedNoDetectionImage(image);
    setViewMode("gallery");
  }, []);

  return (
    <div className="container-fluid py-3 app-shell">
      <div className="row g-3 app-main-row">
        <div className="col-lg-9">
          <div className={viewMode === "map" ? "" : "d-none"}>
            <div className="map-shell border rounded shadow-sm">
              <MapContainer
                crs={L.CRS.EPSG4326}
                bounds={GEO_BOUNDS}
                maxBounds={GEO_BOUNDS}
                maxBoundsViscosity={1.0}
                minZoom={0}
                maxZoom={18}
                zoomSnap={0.1}
                zoomDelta={0.5}
                whenCreated={(mapInstance) => {
                  mapRef.current = mapInstance;
                }}
              >
                <WMSTileLayer
                  url="https://planetarymaps.usgs.gov/cgi-bin/mapserv"
                  map="/maps/earth/moon_simp_cyl.map"
                  service="WMS"
                  version="1.1.1"
                  crs={L.CRS.EPSG4326}
                  layers="KaguyaTC_Ortho"
                  format="image/png"
                  transparent={false}
                  noWrap
                />
                <ZoomOutLevelControl
                  currentLevel={currentLevel}
                  onStepOut={handleStepOutLevel}
                  suppressZoomOutRef={suppressZoomOutRef}
                />
                <FitBoundsOnChange bounds={focusBounds} />
                <HomeControl onHomeClick={handleResetHomeView} />

                {gridCells.map((segment) => {
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
                        weight: isSelected ? 2 : 1,
                        opacity: isSelected ? 0.8 : isHovered ? 0.55 : 0.32,
                        fillColor: color,
                        fillOpacity: isSelected ? 0.14 : isHovered ? 0.08 : 0.04,
                      }}
                      eventHandlers={{
                        mouseover: () => setHoveredSegmentId(segment.id),
                        mouseout: () => setHoveredSegmentId(null),
                        click: () => handleSelectSegment(segment),
                      }}
                    />
                  );
                })}

                {isAnalysisLoading && analysisOverlayBounds && (
                  <Rectangle
                    bounds={analysisOverlayBounds}
                    pathOptions={{
                      color: "#0d6efd",
                      weight: 1,
                      opacity: 0.75,
                      fillColor: "#0d6efd",
                      fillOpacity: 0.24,
                      dashArray: "6 4",
                      interactive: false,
                    }}
                  />
                )}

                {selectedDetection && (
                  <Rectangle
                    key={`overlay-${getDetectionUniqueId(selectedDetection)}`}
                    bounds={detectionToBounds(selectedDetection)}
                    pathOptions={{
                      color: "#fd7e14",
                      weight: 0,
                      fillColor: "#fd7e14",
                      fillOpacity: 0.28,
                      interactive: false,
                    }}
                  />
                )}

                {showBboxes && filteredDetections.map((detection, detectionIndex) => {
                  const detectionUniqueId = getDetectionUniqueId(detection);
                  const detectionRenderKey = `${detectionUniqueId}|${detectionIndex}`;
                  const isSelected = isSameDetection(selectedDetection, detection);
                  const isHovered = hoveredDetectionId === detectionUniqueId;
                  const statusColor = getStatusColor(detection.status);

                  return (
                    <Rectangle
                      key={detectionRenderKey}
                      bounds={detectionToBounds(detection)}
                      pathOptions={{
                        color: statusColor,
                        weight: isSelected ? 5 : isHovered ? 4 : 3,
                        opacity: isSelected ? 0.95 : isHovered ? 0.9 : 0.85,
                        fillColor: statusColor,
                        fillOpacity: isSelected ? 0.3 : isHovered ? 0.2 : 0.14,
                        dashArray: isSelected || isHovered ? null : "5 4",
                      }}
                      eventHandlers={{
                        mouseover: () => setHoveredDetectionId(detectionUniqueId),
                        mouseout: () => setHoveredDetectionId(null),
                        click: () => handleSelectDetection(detection),
                      }}
                    />
                  );
                })}
              </MapContainer>
            </div>
          </div>

          {viewMode === "gallery" && (
            <div className="gallery-shell border rounded shadow-sm p-3">
              <h5 className="mb-1">Przegladarka zdjec</h5>
              {selectedNoDetectionImage ? (
                <div className="mb-3">
                  <div className="d-flex justify-content-between align-items-center gap-2 flex-wrap mb-2">
                    <div className="small text-muted">
                      Podglad wybranego zapisanego zdjecia analizy.
                    </div>
                    <button
                      type="button"
                      className="btn btn-sm btn-outline-secondary"
                      onClick={() => setSelectedNoDetectionImage(null)}
                    >
                      Zamknij podglad
                    </button>
                  </div>
                  <div className="gallery-focus-card border rounded p-2 bg-white">
                    <img
                      src={getAnalysisImageUrl(selectedNoDetectionImage.image_id)}
                      alt={`Zapis analizy ${selectedNoDetectionImage.image_id}`}
                      className="gallery-focus-image"
                    />
                    <div className="small text-muted mt-2">
                      <div><strong>ID:</strong> {selectedNoDetectionImage.image_id}</div>
                      <div><strong>Status:</strong> {selectedNoDetectionImage.status || NO_DETECTIONS_FILTER}</div>
                      <div>
                        <strong>Lat/Lon:</strong>{" "}
                        {typeof selectedNoDetectionImage.lat === "number"
                          ? selectedNoDetectionImage.lat.toFixed(6)
                          : "-"}
                        {" / "}
                        {typeof selectedNoDetectionImage.lon === "number"
                          ? selectedNoDetectionImage.lon.toFixed(6)
                          : "-"}
                      </div>
                      <div><strong>Rozdzielczosc:</strong> {selectedNoDetectionImage.resolution || "-"}</div>
                      {selectedNoDetectionImage.timestamp && (
                        <div><strong>Czas:</strong> {selectedNoDetectionImage.timestamp}</div>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="small text-muted mb-3">
                  {isNoDetectionsFilterSelected
                    ? "Widok kart oparty o zapisane obrazy analizy."
                    : "Widok kart oparty o aktualnie widoczne detekcje (filtr statusu jest zachowany)."}
                </div>
              )}

              {selectedNoDetectionImage ? null : isNoDetectionsFilterSelected ? (
                visibleAnalysisImages.length === 0 ? (
                  <div className="small text-muted">
                    {analysisImagesFilter === ALL_ANALYSIS_IMAGES_FILTER
                      ? "Brak zapisanych obrazow analizy."
                      : "Brak zapisanych obrazow z wynikiem no_detections."}
                  </div>
                ) : (
                  <div className="row g-3">
                    {visibleAnalysisImages.map((image, imageIndex) => {
                      const itemKey = `${image.image_id || "no-id"}|${image.timestamp || "no-ts"}|${imageIndex}`;
                      const isNoDetectionsStatus =
                        (typeof image.status === "string" ? image.status : NO_DETECTIONS_FILTER) ===
                        NO_DETECTIONS_FILTER;

                      return (
                        <div className="col-sm-6 col-xl-4" key={`gallery-image-${itemKey}`}>
                          <button
                            type="button"
                            className="card h-100 shadow-sm border-0 text-start w-100 p-0"
                            onClick={() => handleOpenAnalysisImage(image)}
                          >
                            <img
                              src={getAnalysisImageUrl(image.image_id)}
                              alt={`Zapis analizy ${image.image_id}`}
                              loading="lazy"
                              className="gallery-preview card-img-top"
                            />
                            <div className="card-body py-2">
                              <div className="fw-semibold small">{image.image_id || "analysis_image"}</div>
                              <div className="small mt-1">
                                <span
                                  className={`badge ${isNoDetectionsStatus ? "text-bg-secondary" : "text-bg-primary"}`}
                                >
                                  {image.status || NO_DETECTIONS_FILTER}
                                </span>
                              </div>
                              <div className="small text-muted">
                                rozdzielczosc: {image.resolution || "-"}
                              </div>
                            </div>
                          </button>
                        </div>
                      );
                    })}
                  </div>
                )
              ) : detections.length === 0 ? (
                <div className="small text-muted">Brak detekcji. Kliknij "Uruchom analize".</div>
              ) : filteredDetections.length === 0 ? (
                <div className="small text-muted">Brak detekcji dla statusu: {statusFilter}.</div>
              ) : (
                <div className="row g-3">
                  {filteredDetections.map((detection, detectionIndex) => {
                    const detectionUniqueId = getDetectionUniqueId(detection);
                    const detectionRenderKey = `${detectionUniqueId}|${detectionIndex}`;
                    const statusBadgeClass = getStatusBadgeClass(detection.status);
                    const previewUrl = getDetectionPreviewUrl(detection);

                    return (
                      <div className="col-sm-6 col-xl-4" key={`gallery-${detectionRenderKey}`}>
                        <div className="card h-100 shadow-sm border-0">
                          {previewUrl ? (
                            <img
                              src={previewUrl}
                              alt={`Podglad detekcji ${detection.detection_id}`}
                              loading="lazy"
                              className="gallery-preview card-img-top"
                            />
                          ) : (
                            <div className="gallery-preview-fallback card-img-top">Brak podgladu</div>
                          )}
                          <div className="card-body py-2">
                            <div className="fw-semibold small">{detection.detection_id}</div>
                            <div className="small mt-1">
                              <span className={`badge ${statusBadgeClass}`}>{detection.status}</span>
                            </div>
                            <div className="small text-muted">
                              confidence: {Number(detection.confidence).toFixed(2)}
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="col-lg-3 app-sidebar-column">
          <div className="card shadow-sm sidebar-card">
            <div className="card-body sidebar-card-body">
              <h5 className="card-title">Panel obszaru</h5>

              <div className="small text-muted mb-2">Widok aplikacji</div>
              <div className="btn-group btn-group-sm w-100 mb-3" role="group" aria-label="Tryb widoku aplikacji">
                <button
                  type="button"
                  className={`btn ${viewMode === "map" ? "btn-primary" : "btn-outline-primary"}`}
                  onClick={() => setViewMode("map")}
                >
                  Mapa
                </button>
                <button
                  type="button"
                  className={`btn ${viewMode === "gallery" ? "btn-primary" : "btn-outline-primary"}`}
                  onClick={() => setViewMode("gallery")}
                >
                  Przegladarka zdjec
                </button>
              </div>

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

              <button
                className="btn btn-primary w-100 mb-2 d-flex align-items-center justify-content-center gap-2"
                onClick={handleChooseArea}
                disabled={isLoadingDetections}
              >
                {isLoadingDetections && (
                  <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true" />
                )}
                <span>{isLoadingDetections ? "Analizowanie..." : "Uruchom analize"}</span>
              </button>

              <button
                className="btn btn-outline-primary w-100 mb-2 d-flex align-items-center justify-content-center gap-2"
                onClick={handleLocalAnalysis}
                disabled={isLoadingDetections}
              >
                {isLoadingDetections && (
                  <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true" />
                )}
                <span>{isLoadingDetections ? "Analizowanie..." : "Analiza lokalna"}</span>
              </button>

              <button className="btn btn-outline-secondary w-100 mb-2" onClick={handleResetHomeView}>
                Reset widoku
              </button>

              <div className="form-check form-switch mb-2">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="toggle-lock-level"
                  checked={isLevelLocked}
                  onChange={(event) => setIsLevelLocked(event.target.checked)}
                />
                <label className="form-check-label" htmlFor="toggle-lock-level">
                  Lock level
                </label>
              </div>

              <div className="small text-muted mb-2">
                {isLevelLocked ? "Klik tylko zaznacza segment." : "Klik schodzi poziom nizej."}
              </div>

              <div className="small text-muted mb-3">Poziom siatki: {currentLevel}</div>

              {analysisStatus && (
                <div className="small mb-3">
                  Status analizy:{" "}
                  <span
                    className={`badge ${
                      analysisStatus === "loading"
                        ? "text-bg-info"
                        : analysisStatus === "success"
                          ? "text-bg-success"
                          : "text-bg-danger"
                    }`}
                  >
                    {analysisStatus}
                  </span>
                </div>
              )}

              {currentAnalysisId && (
                <div className="small text-muted mb-3">analysis_id: {currentAnalysisId}</div>
              )}

              <div className="small text-muted mb-2">Rozdzielczosc analizy</div>
              <div className="btn-group btn-group-sm w-100 mb-2" role="group" aria-label="Tryb rozdzielczosci analizy">
                <button
                  type="button"
                  className={`btn ${resolutionMode === "preview" ? "btn-primary" : "btn-outline-primary"}`}
                  onClick={() => setResolutionMode("preview")}
                >
                  preview
                </button>
                <button
                  type="button"
                  className={`btn ${resolutionMode === "detail" ? "btn-primary" : "btn-outline-primary"}`}
                  onClick={() => setResolutionMode("detail")}
                >
                  detail
                </button>
                <button
                  type="button"
                  className={`btn ${resolutionMode === "ultra" ? "btn-primary" : "btn-outline-primary"}`}
                  onClick={() => setResolutionMode("ultra")}
                >
                  ultra
                </button>
              </div>
              <div className="small text-muted mb-3">
                {RESOLUTION_DESCRIPTION_MAP[resolutionMode]} (ok. {RESOLUTION_MPP_MAP[resolutionMode].toFixed(2)} mpp)
              </div>

              <div className="small text-muted mb-2">Zrodlo WMS</div>
              <div className="btn-group btn-group-sm w-100 mb-3" role="group" aria-label="Zrodlo WMS">
                {WMS_SOURCE_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    className={`btn ${wmsSource === option.value ? "btn-primary" : "btn-outline-primary"}`}
                    onClick={() => setWmsSource(option.value)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>

              <div className="small text-muted mb-2">Warstwa WMS</div>
              <div className="btn-group btn-group-sm w-100 mb-3" role="group" aria-label="Warstwa WMS">
                {selectedWmsLayerOptions.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    className={`btn ${wmsLayer === option.value ? "btn-primary" : "btn-outline-primary"}`}
                    onClick={() => setWmsLayer(option.value)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
              {isGlobalMosaicLayerSelected && (
                <div className="alert alert-warning py-2 small mb-3">
                  Uwaga: ta warstwa to globalna mozaika o nizszym detalu. Dla trudnych obiektow moze nie byc detekcji.
                </div>
              )}

              <div className="small text-muted mb-2">Liczba probek</div>
              <div className="btn-group btn-group-sm w-100 mb-2" role="group" aria-label="Liczba probek analizy">
                <button
                  type="button"
                  className={`btn ${numSamples === 1 ? "btn-primary" : "btn-outline-primary"}`}
                  onClick={() => setNumSamples(1)}
                >
                  1
                </button>
                <button
                  type="button"
                  className={`btn ${numSamples === 5 ? "btn-primary" : "btn-outline-primary"}`}
                  onClick={() => setNumSamples(5)}
                >
                  5
                </button>
                <button
                  type="button"
                  className={`btn ${numSamples === 10 ? "btn-primary" : "btn-outline-primary"}`}
                  onClick={() => setNumSamples(10)}
                >
                  10
                </button>
              </div>
              <div className="small text-muted mb-3">Wiecej probek = dluzsza analiza</div>

              <div className="small text-muted mb-2">Confidence threshold ({confidenceThreshold.toFixed(2)})</div>
              <input
                className="form-range mb-3"
                type="range"
                min="0.01"
                max="1.0"
                step="0.01"
                value={confidenceThreshold}
                onChange={(event) => setConfidenceThreshold(Number(event.target.value))}
              />

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

              <div className="form-check form-switch mt-3">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="toggle-bboxes"
                  checked={showBboxes}
                  onChange={(event) => setShowBboxes(event.target.checked)}
                />
                <label className="form-check-label" htmlFor="toggle-bboxes">
                  Pokaz bounding boxy
                </label>
              </div>

              {chosenMessage && (
                <div
                  className={`alert ${isNoCoverageChosenMessage ? "alert-warning" : "alert-info"} py-2 mt-3 mb-0`}
                >
                  {chosenMessage}
                </div>
              )}

              <hr className="my-4" />
              <h6 className="mb-3">Detekcje ({detectionSectionCount})</h6>

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
                <button
                  type="button"
                  className={`btn ${statusFilter === NO_DETECTIONS_FILTER ? "btn-secondary" : "btn-outline-secondary"}`}
                  onClick={() => setStatusFilter(NO_DETECTIONS_FILTER)}
                >
                  no_detections
                </button>
              </div>

              <div className="detection-list-scroll" ref={detectionListRef}>
                {isNoDetectionsFilterSelected ? (
                  <>
                    <div className="small text-muted mb-2">Zakres zapisanych obrazow</div>
                    <div className="btn-group btn-group-sm w-100 mb-3" role="group" aria-label="Zakres zapisanych obrazow">
                      <button
                        type="button"
                        className={`btn ${analysisImagesFilter === NO_DETECTIONS_FILTER ? "btn-secondary" : "btn-outline-secondary"}`}
                        onClick={() => setAnalysisImagesFilter(NO_DETECTIONS_FILTER)}
                      >
                        no_detections
                      </button>
                      <button
                        type="button"
                        className={`btn ${analysisImagesFilter === ALL_ANALYSIS_IMAGES_FILTER ? "btn-secondary" : "btn-outline-secondary"}`}
                        onClick={() => setAnalysisImagesFilter(ALL_ANALYSIS_IMAGES_FILTER)}
                      >
                        wszystkie
                      </button>
                    </div>

                    {visibleAnalysisImages.length === 0 ? (
                      <div className="small text-muted">
                        {analysisImagesFilter === ALL_ANALYSIS_IMAGES_FILTER
                          ? "Brak zapisanych obrazow analizy."
                          : "Brak przeanalizowanych zdjec z wynikiem no_detections."}
                      </div>
                    ) : (
                      <div className="list-group">
                        {groupedAnalysisImages.map((group) => (
                          <div key={`group-${group.analysisId}`} className="border-bottom pb-2 mb-2">
                            <div className="small fw-semibold text-muted px-1 mb-1">
                              sesja: {group.analysisId} ({group.images.length})
                            </div>

                            <div className="list-group">
                              {group.images.map((image, imageIndex) => {
                                const itemKey = `${image.image_id || "no-id"}|${image.timestamp || "no-ts"}|${imageIndex}`;
                                const imageName = getFileNameFromPath(image.path);
                                const isActive = selectedNoDetectionImage?.image_id === image.image_id;

                                return (
                                  <button
                                    type="button"
                                    key={itemKey}
                                    className={`list-group-item list-group-item-action text-start ${
                                      isActive ? "active" : ""
                                    }`}
                                    onClick={() => handleOpenAnalysisImage(image)}
                                  >
                                    <div><strong>{image.image_id || "analysis_image"}</strong></div>
                                    {imageName && <div className={isActive ? "small text-white-50" : "small text-muted"}>plik: {imageName}</div>}
                                    <div className={isActive ? "small text-white-50" : "small text-muted"}>status: {image.status || NO_DETECTIONS_FILTER}</div>
                                    <div className={isActive ? "small text-white-50" : "small text-muted"}>rozdzielczosc: {image.resolution || "-"}</div>
                                    <div className={isActive ? "small text-white-50" : "small text-muted"}>
                                      lat: {typeof image.lat === "number" ? image.lat.toFixed(6) : "-"}, lon:{" "}
                                      {typeof image.lon === "number" ? image.lon.toFixed(6) : "-"}
                                    </div>
                                    {image.timestamp && (
                                      <div className={isActive ? "small text-white-50" : "small text-muted"}>czas: {image.timestamp}</div>
                                    )}
                                  </button>
                                );
                              })}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                ) : detections.length === 0 ? (
                  <div className="small text-muted">Brak detekcji. Kliknij "Uruchom analize".</div>
                ) : filteredDetections.length === 0 ? (
                  <div className="small text-muted">Brak detekcji dla statusu: {statusFilter}.</div>
                ) : (
                  <div className="list-group">
                    {filteredDetections.map((detection, detectionIndex) => {
                      const detectionUniqueId = getDetectionUniqueId(detection);
                      const detectionRenderKey = `${detectionUniqueId}|${detectionIndex}`;
                      const isSelected = isSameDetection(selectedDetection, detection);
                      const isHovered = hoveredDetectionId === detectionUniqueId;
                      const statusBadgeClass = getStatusBadgeClass(detection.status);
                      const commentText = (detection.comment ?? "").trim();
                      const hasComment = commentText.length > 0;
                      const isEditingThis = editingDetectionId === detection.detection_id;
                      const isInputVisible = isEditingThis || (!hasComment && isSelected);

                      return (
                        <div
                          key={detectionRenderKey}
                          ref={(node) => {
                            if (node) {
                              detectionItemRefs.current.set(detectionRenderKey, node);
                            } else {
                              detectionItemRefs.current.delete(detectionRenderKey);
                            }
                          }}
                          onMouseEnter={() => setHoveredDetectionId(detectionUniqueId)}
                          onMouseLeave={() => setHoveredDetectionId(null)}
                          className={`list-group-item list-group-item-action text-start ${
                            isSelected ? "bg-primary-subtle border-primary" : ""
                          }`}
                          style={
                            !isSelected && isHovered
                              ? {
                                  backgroundColor: "rgba(255, 193, 7, 0.08)",
                                  boxShadow: "inset 0 0 0 1px rgba(255, 193, 7, 0.55)",
                                }
                              : undefined
                          }
                        >
                          <button
                            type="button"
                            onClick={() => handleSelectDetection(detection)}
                            className="btn btn-link text-decoration-none text-reset p-0 w-100 text-start"
                          >
                            <div><strong>{detection.detection_id}</strong></div>
                            <div className="small mt-1">
                              <span className={`badge ${statusBadgeClass}`}>{detection.status}</span>
                            </div>
                            <div className="small text-muted">
                              confidence: {Number(detection.confidence).toFixed(2)}
                            </div>
                          </button>

                          {hasComment && (
                            <div className="small border rounded bg-light px-2 py-1 mt-2 d-flex align-items-start justify-content-between gap-2">
                              <div className="flex-grow-1">
                                <strong>Komentarz:</strong> {commentText}
                              </div>
                              <div className="d-flex gap-1">
                                {!isEditingThis && (
                                  <button
                                    type="button"
                                    className="btn btn-sm btn-outline-secondary px-2 py-0"
                                    onClick={() => handleStartEditComment(detection)}
                                    title="Edytuj komentarz"
                                    aria-label="Edytuj komentarz"
                                  >
                                    <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true" focusable="false">
                                      <path
                                        fill="currentColor"
                                        d="M12.854 1.146a.5.5 0 0 1 0 .708L6.207 8.5H4v-2.207l6.646-6.647a.5.5 0 0 1 .708 0l1.5 1.5zm-8.354 8.354L10.646 3.354l2 2L6.5 11.5H4.5v-2zM2 13h12v1H2v-1z"
                                      />
                                    </svg>
                                  </button>
                                )}
                                <button
                                  type="button"
                                  className="btn btn-sm btn-outline-danger px-2 py-0"
                                  onClick={() => handleDeleteComment(detection.detection_id)}
                                  title="Usun komentarz"
                                  aria-label="Usun komentarz"
                                >
                                  <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true" focusable="false">
                                    <path
                                      fill="currentColor"
                                      d="M5.5 5.5a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm5 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6zM1 3.5A.5.5 0 0 1 1.5 3H4V2a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v1h2.5a.5.5 0 0 1 0 1h-.538l-.853 10.66A1 1 0 0 1 12.112 15H3.888a1 1 0 0 1-.997-.84L2.038 4.5H1.5a.5.5 0 0 1-.5-.5zM5 2v1h6V2H5z"
                                    />
                                  </svg>
                                </button>
                              </div>
                            </div>
                          )}

                          <div className="d-flex gap-2 mt-2 flex-wrap">
                            {hasComment ? (
                              null
                            ) : (
                              !isInputVisible && (
                                <button
                                  type="button"
                                  className="btn btn-sm btn-outline-primary"
                                  onClick={() => {
                                    setSelectedDetection(detection);
                                    setEditingDetectionId(null);
                                    setInputComment("");
                                  }}
                                >
                                  Dodaj komentarz
                                </button>
                              )
                            )}
                          </div>

                          {isInputVisible && (
                            <div className="d-flex gap-2 mt-2">
                              <input
                                className="form-control form-control-sm"
                                type="text"
                                placeholder="Wpisz komentarz"
                                value={inputComment}
                                onChange={(event) => setInputComment(event.target.value)}
                              />
                              <button
                                type="button"
                                className="btn btn-sm btn-outline-primary"
                                onClick={() => handleSaveComment(detection.detection_id)}
                              >
                                {isEditingThis ? "Zapisz zmiany" : "Zapisz"}
                              </button>
                              {isEditingThis && (
                                <button
                                  type="button"
                                  className="btn btn-sm btn-outline-secondary"
                                  onClick={() => {
                                    setEditingDetectionId(null);
                                    setInputComment("");
                                  }}
                                >
                                  Anuluj
                                </button>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
