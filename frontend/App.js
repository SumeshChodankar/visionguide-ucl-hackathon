/**
 * VisionGuide Accessible Mobile Client (frontend/App.js)
 * Built with React Native (Expo SDK 51+):
 * - expo-camera: Continuous 3.5-second frame capture for parallel multi-agent vision
 * - expo-location: Live outdoor GPS telemetry for Google Maps walking navigation
 * - expo-speech: Low-latency local TTS with immediate HIGH-hazard preemption
 * - expo-av: Giant bottom-screen Push-to-Talk voice command recorder (Whisper-1)
 * - expo-haptics: Distinct tactile vibration patterns for hazards, warnings, and arrival
 */

import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  StyleSheet,
  Text,
  View,
  Pressable,
  SafeAreaView,
  ScrollView,
  Platform,
  ActivityIndicator,
} from "react-native";
import { StatusBar } from "expo-status-bar";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as Location from "expo-location";
import * as Speech from "expo-speech";
import { Audio } from "expo-av";
import * as Haptics from "expo-haptics";

const API_BASE_URL =
  process.env.EXPO_PUBLIC_API_BASE_URL || "http://192.168.1.100:8000";
const SESSION_ID = "visionguide-mobile-user-01";
const FRAME_INTERVAL_MS = 3500;

export default function App() {
  // Permissions & Hardware Refs
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [hasAudioPermission, setHasAudioPermission] = useState(false);
  const [hasLocationPermission, setHasLocationPermission] = useState(false);
  const cameraRef = useRef(null);
  const recordingRef = useRef(null);
  const isAnalyzingRef = useRef(false);
  const lastSpokenRef = useRef("");

  // Navigation & Multi-Agent Telemetry State
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessingFrame, setIsProcessingFrame] = useState(false);
  const [autoScanEnabled, setAutoScanEnabled] = useState(true);
  const [coords, setCoords] = useState({ latitude: 51.5246, longitude: -0.134 });

  const [navMode, setNavMode] = useState("IDLE");
  const [activeGoal, setActiveGoal] = useState(null);
  const [destinationBuilding, setDestinationBuilding] = useState(null);
  const [targetRoom, setTargetRoom] = useState(null);
  const [targetFloor, setTargetFloor] = useState(null);
  const [currentFloor, setCurrentFloor] = useState(1);
  const [floorMismatch, setFloorMismatch] = useState(false);
  const [arrived, setArrived] = useState(false);

  const [hazardPriority, setHazardPriority] = useState("NONE");
  const [spokenGuidance, setSpokenGuidance] = useState(
    "Welcome to VisionGuide. Hold the giant bottom button and speak your destination, for example: Take me to the Engineering Building, Room 443B."
  );
  const [hazardsList, setHazardsList] = useState([]);
  const [ocrSignsList, setOcrSignsList] = useState([]);
  const [latencyMs, setLatencyMs] = useState(0);

  // ============================================================================
  // ACCESSIBLE SPEECH & HAPTICS ENGINE
  // ============================================================================

  const speakGuidance = useCallback(
    async (text, interrupt = false, priority = "NONE") => {
      if (!text) return;

      // Avoid repeating identical non-urgent guidance every 3.5s
      if (!interrupt && priority !== "HIGH" && text === lastSpokenRef.current) {
        return;
      }
      lastSpokenRef.current = text;

      try {
        const speaking = await Speech.isSpeakingAsync();
        if (interrupt || priority === "HIGH") {
          if (speaking) {
            await Speech.stop();
          }
          Speech.speak(text, {
            language: "en-US",
            pitch: 1.08,
            rate: 1.06,
          });
        } else if (!speaking) {
          Speech.speak(text, {
            language: "en-US",
            pitch: 1.0,
            rate: 0.98,
          });
        }
      } catch (err) {
        console.warn("TTS error:", err);
      }
    },
    []
  );

  const triggerHapticPattern = useCallback(async (pattern, priority) => {
    try {
      if (pattern === "URGENT_PULSE" || priority === "HIGH") {
        // Rapid triple heavy impact for immediate obstacle preemption
        await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
        setTimeout(() => {
          Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
        }, 160);
        setTimeout(() => {
          Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
        }, 320);
      } else if (pattern === "WARNING_DOUBLE_TAP" || priority === "MEDIUM") {
        await Haptics.notificationAsync(
          Haptics.NotificationFeedbackType.Warning
        );
      } else if (pattern === "ARRIVAL_SUCCESS") {
        await Haptics.notificationAsync(
          Haptics.NotificationFeedbackType.Success
        );
      } else if (pattern === "COMMAND_ACCEPTED") {
        await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      }
    } catch (err) {
      console.warn("Haptics error:", err);
    }
  }, []);

  // ============================================================================
  // INITIALIZE PERMISSIONS & CONTINUOUS GPS WATCHER
  // ============================================================================

  useEffect(() => {
    let locationSubscription = null;

    (async () => {
      if (!cameraPermission?.granted) {
        await requestCameraPermission();
      }

      const audioPerm = await Audio.requestPermissionsAsync();
      setHasAudioPermission(audioPerm.status === "granted");

      const locPerm = await Location.requestForegroundPermissionsAsync();
      if (locPerm.status === "granted") {
        setHasLocationPermission(true);
        locationSubscription = await Location.watchPositionAsync(
          {
            accuracy: Location.Accuracy.BestForNavigation,
            timeInterval: 3000,
            distanceInterval: 2,
          },
          (loc) => {
            setCoords({
              latitude: loc.coords.latitude,
              longitude: loc.coords.longitude,
            });
          }
        );
      }

      // Initial welcome announcement
      speakGuidance(
        "VisionGuide ready. Hold the bottom screen button to speak your destination.",
        true,
        "NONE"
      );
    })();

    return () => {
      if (locationSubscription) {
        locationSubscription.remove();
      }
      Speech.stop();
    };
  }, []);

  // ============================================================================
  // APPLY BACKEND TELEMETRY TO HUD, HAPTICS, AND SPEECH
  // ============================================================================

  const applyBackendResponse = useCallback(
    (data, forceInterrupt = false) => {
      if (!data) return;

      const priority = data.hazard_priority || "NONE";
      const interrupt = Boolean(data.interrupt_speech) || forceInterrupt;
      const guidance = data.spoken_guidance || "";

      setHazardPriority(priority);
      if (guidance) {
        setSpokenGuidance(guidance);
      }
      if (data.navigation_mode) setNavMode(data.navigation_mode);
      if (data.active_goal !== undefined) setActiveGoal(data.active_goal);
      if (data.destination_building !== undefined) {
        setDestinationBuilding(data.destination_building);
      }
      if (data.target_room !== undefined) setTargetRoom(data.target_room);
      if (data.target_floor !== undefined) setTargetFloor(data.target_floor);
      if (data.current_floor !== undefined && data.current_floor !== null) {
        setCurrentFloor(data.current_floor);
      }
      if (data.floor_mismatch !== undefined) {
        setFloorMismatch(Boolean(data.floor_mismatch));
      }
      if (data.arrived !== undefined) {
        setArrived(Boolean(data.arrived));
      }
      if (data.total_latency_ms !== undefined) {
        setLatencyMs(Math.round(data.total_latency_ms));
      } else if (data.latency_ms !== undefined) {
        setLatencyMs(Math.round(data.latency_ms));
      }

      if (data.agents) {
        setHazardsList(data.agents.hazard?.hazards || []);
        setOcrSignsList(data.agents.ocr?.texts || []);
      }

      triggerHapticPattern(data.haptic_pattern, priority);
      speakGuidance(guidance, interrupt, priority);
    },
    [speakGuidance, triggerHapticPattern]
  );

  // ============================================================================
  // 3.5-SECOND AUTOMATIC CAMERA FRAME ANALYSIS LOOP
  // ============================================================================

  const captureAndAnalyzeFrame = useCallback(
    async (simulatedOverrides = null) => {
      if (isAnalyzingRef.current || isRecording) return;
      isAnalyzingRef.current = true;
      setIsProcessingFrame(true);

      try {
        let base64Image = "";
        if (cameraRef.current && cameraPermission?.granted && !simulatedOverrides) {
          const photo = await cameraRef.current.takePictureAsync({
            quality: 0.45,
            base64: true,
            skipProcessing: true,
            shutterSound: false,
          });
          base64Image = photo?.base64 || "";
        }

        const formData = new FormData();
        formData.append("session_id", SESSION_ID);
        formData.append("lat", String(coords.latitude));
        formData.append("lng", String(coords.longitude));
        if (base64Image) {
          formData.append("frame_base64", base64Image);
        }
        if (simulatedOverrides) {
          formData.append(
            "simulated_overrides_json",
            JSON.stringify(simulatedOverrides)
          );
        }

        const response = await fetch(`${API_BASE_URL}/api/v1/analyze-frame`, {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          applyBackendResponse(result, Boolean(simulatedOverrides));
        }
      } catch (err) {
        console.warn("Frame analysis network warning:", err.message);
      } finally {
        isAnalyzingRef.current = false;
        setIsProcessingFrame(false);
      }
    },
    [cameraPermission, coords, isRecording, applyBackendResponse]
  );

  useEffect(() => {
    if (!autoScanEnabled) return;
    const intervalId = setInterval(() => {
      captureAndAnalyzeFrame(null);
    }, FRAME_INTERVAL_MS);

    return () => clearInterval(intervalId);
  }, [autoScanEnabled, captureAndAnalyzeFrame]);

  // ============================================================================
  // PUSH-TO-TALK VOICE COMMAND HANDLERS (EXPO-AV -> WHISPER-1)
  // ============================================================================

  const startPushToTalk = async () => {
    try {
      await Speech.stop();
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);

      if (!hasAudioPermission) {
        const perm = await Audio.requestPermissionsAsync();
        if (perm.status !== "granted") {
          speakGuidance(
            "Microphone permission is required for voice commands.",
            true,
            "NONE"
          );
          return;
        }
        setHasAudioPermission(true);
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      recordingRef.current = recording;
      setIsRecording(true);
    } catch (err) {
      console.error("Failed to start recording:", err);
      setIsRecording(false);
    }
  };

  const stopPushToTalk = async () => {
    if (!recordingRef.current) {
      setIsRecording(false);
      return;
    }

    try {
      setIsRecording(false);
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      await recordingRef.current.stopAndUnloadAsync();
      const uri = recordingRef.current.getURI();
      recordingRef.current = null;

      if (!uri) return;

      const formData = new FormData();
      formData.append("session_id", SESSION_ID);
      formData.append("lat", String(coords.latitude));
      formData.append("lng", String(coords.longitude));
      formData.append("audio", {
        uri,
        name: "voice_command.m4a",
        type: "audio/m4a",
      });

      const response = await fetch(`${API_BASE_URL}/api/v1/voice-command`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        if (data.session) {
          setActiveGoal(data.session.active_goal);
          setDestinationBuilding(data.session.destination_building);
          setTargetRoom(data.session.target_room);
          setTargetFloor(data.session.target_floor);
          setCurrentFloor(data.session.current_floor ?? 1);
          setNavMode(data.session.navigation_mode || "OUTDOOR");
          setArrived(Boolean(data.session.arrived));
        }
        applyBackendResponse(data, true);
      }
    } catch (err) {
      console.error("Voice command submission error:", err);
      speakGuidance(
        "Unable to reach VisionGuide server. Check your backend connection.",
        true,
        "NONE"
      );
    }
  };

  // Helper for instant accessible preset commands / demo verification
  const sendPresetVoiceCommand = async (transcriptText) => {
    try {
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      const formData = new FormData();
      formData.append("session_id", SESSION_ID);
      formData.append("lat", String(coords.latitude));
      formData.append("lng", String(coords.longitude));
      formData.append("transcript_override", transcriptText);

      const response = await fetch(`${API_BASE_URL}/api/v1/voice-command`, {
        method: "POST",
        body: formData,
      });
      if (response.ok) {
        const data = await response.json();
        if (data.session) {
          setActiveGoal(data.session.active_goal);
          setDestinationBuilding(data.session.destination_building);
          setTargetRoom(data.session.target_room);
          setTargetFloor(data.session.target_floor);
          setCurrentFloor(data.session.current_floor ?? 1);
          setNavMode(data.session.navigation_mode || "OUTDOOR");
          setArrived(Boolean(data.session.arrived));
        }
        applyBackendResponse(data, true);
      }
    } catch (err) {
      console.warn("Preset voice command failed:", err.message);
    }
  };

  // ============================================================================
  // HIGH-CONTRAST ACCESSIBLE HUD RENDERING
  // ============================================================================

  const getPriorityBannerStyle = () => {
    if (hazardPriority === "HIGH") return styles.bannerHigh;
    if (hazardPriority === "MEDIUM") return styles.bannerMedium;
    if (arrived) return styles.bannerArrived;
    return styles.bannerSafe;
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="light" />

      {/* Live Camera Preview Layer with High-Contrast Dark Scrim */}
      <View style={styles.cameraContainer}>
        {cameraPermission?.granted ? (
          <CameraView
            ref={cameraRef}
            style={StyleSheet.absoluteFillObject}
            facing="back"
          />
        ) : (
          <View style={styles.cameraFallback}>
            <Text style={styles.cameraFallbackText}>
              CAMERA PERMISSION REQUIRED FOR LIVE OBSTACLE & OCR DETECTION
            </Text>
          </View>
        )}
        <View style={styles.cameraScrim} />

        {/* TOP HUD: STATUS & HAZARD PRIORITY BANNER */}
        <View
          style={[styles.priorityBanner, getPriorityBannerStyle()]}
          accessible={true}
          accessibilityRole="alert"
          accessibilityLiveRegion="assertive"
        >
          <Text style={styles.priorityTitle}>
            {hazardPriority === "HIGH"
              ? "URGENT HAZARD IN PATH"
              : hazardPriority === "MEDIUM"
              ? "CAUTION: OBSTACLE NEARBY"
              : arrived
              ? "DESTINATION ARRIVED"
              : `MODE: ${navMode} • SCAN ${autoScanEnabled ? "3.5S ON" : "PAUSED"}`}
          </Text>
          <Text style={styles.latencyBadge}>
            {isProcessingFrame ? "SCANNING..." : `${latencyMs} ms`}
          </Text>
        </View>

        {/* FLOOR & GOAL TELEMETRY BAR */}
        <View style={styles.telemetryGrid} accessible={true}>
          <View style={styles.telemetryCard}>
            <Text style={styles.telemetryLabel}>ACTIVE DESTINATION</Text>
            <Text style={styles.telemetryValue} numberOfLines={1}>
              {destinationBuilding || activeGoal || "None (Hold PTT below)"}
            </Text>
          </View>
          <View style={styles.telemetryRow}>
            <View style={[styles.telemetryPill, styles.pillCyan]}>
              <Text style={styles.pillLabel}>TARGET ROOM</Text>
              <Text style={styles.pillValue}>{targetRoom || "--"}</Text>
            </View>
            <View
              style={[
                styles.telemetryPill,
                floorMismatch ? styles.pillWarning : styles.pillYellow,
              ]}
            >
              <Text style={styles.pillLabel}>FLOOR (NOW / GOAL)</Text>
              <Text style={styles.pillValue}>
                F{currentFloor} /{" "}
                {targetFloor !== null && targetFloor !== undefined
                  ? `F${targetFloor}`
                  : "--"}
              </Text>
            </View>
          </View>
        </View>

        {/* PRIMARY SPOKEN GUIDANCE CAPTION BOX */}
        <Pressable
          style={styles.guidanceBox}
          onPress={() => speakGuidance(spokenGuidance, true, hazardPriority)}
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel={`Spoken guidance: ${spokenGuidance}. Double tap to repeat audio.`}
        >
          <Text style={styles.guidanceHeader}>
            LIVE AUDIO GUIDANCE (TAP TO REPEAT)
          </Text>
          <Text style={styles.guidanceText}>{spokenGuidance}</Text>
        </Pressable>

        {/* DETECTED CLOCK-FACE HAZARDS & OCR SIGNS STRIP */}
        <View style={styles.detectionsStrip}>
          <Text style={styles.stripTitle}>
            YOLO HAZARDS:{" "}
            {hazardsList.length > 0
              ? hazardsList
                  .slice(0, 2)
                  .map(
                    (h) =>
                      `${h.object.toUpperCase()} @ ${h.clock_direction} (${
                        h.distance_meters
                      }m)`
                  )
                  .join(" | ")
              : "CLEAR (10–2 O'CLOCK)"}
          </Text>
          <Text style={styles.stripSub}>
            EASYOCR SIGNS:{" "}
            {ocrSignsList.length > 0
              ? ocrSignsList
                  .slice(0, 3)
                  .map((s) => `"${s.text}" [${s.spatial_tag.toUpperCase()}]`)
                  .join(" • ")
              : "SCANNING PLAQUES & FLOOR INDICATORS..."}
          </Text>
        </View>

        {/* QUICK DEMO / ACCESSIBILITY SCENARIO TRIGGERS */}
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.demoStrip}
        >
          <Pressable
            style={styles.demoChip}
            onPress={() =>
              sendPresetVoiceCommand(
                "Take me to UCL Engineering Building, Room 443B"
              )
            }
            accessibilityRole="button"
            accessibilityLabel="Set destination to UCL Engineering Building Room 443B"
          >
            <Text style={styles.demoChipText}>1. Goal: Room 443B</Text>
          </Pressable>

          <Pressable
            style={styles.demoChip}
            onPress={() =>
              captureAndAnalyzeFrame({
                ocr_items: [
                  { text: "Floor 1", confidence: 0.96, x_center_norm: 0.25 },
                  {
                    text: "Elevators ->",
                    confidence: 0.94,
                    x_center_norm: 0.72,
                  },
                ],
                vision_result: {
                  wet_floor_or_spill: false,
                  glass_entrance_detected: true,
                  elevator_detected: true,
                  elevator_direction: "right",
                  detected_floor_number: 1,
                  scene_summary: "Building lobby on Floor 1 with elevators on right.",
                },
              })
            }
            accessibilityRole="button"
            accessibilityLabel="Simulate entering Floor 1 lobby when target is Floor 4"
          >
            <Text style={styles.demoChipText}>2. Sim: Floor 1 Mismatch</Text>
          </Pressable>

          <Pressable
            style={[styles.demoChip, styles.demoChipDanger]}
            onPress={() =>
              captureAndAnalyzeFrame({
                yolo_detections: [
                  {
                    class_name: "chair",
                    confidence: 0.92,
                    bbox: [240, 220, 400, 465],
                  },
                ],
                vision_result: {
                  wet_floor_or_spill: true,
                  surface_hazard_description: "Yellow wet floor cone and water spill",
                  scene_summary: "Obstacle and wet floor directly ahead.",
                },
              })
            }
            accessibilityRole="button"
            accessibilityLabel="Simulate urgent hazard at 12 o'clock"
          >
            <Text style={styles.demoChipText}>3. Sim: HIGH Hazard</Text>
          </Pressable>

          <Pressable
            style={[styles.demoChip, styles.demoChipSuccess]}
            onPress={() =>
              captureAndAnalyzeFrame({
                ocr_items: [
                  { text: "Floor 4", confidence: 0.98, x_center_norm: 0.2 },
                  { text: "Room 443B", confidence: 0.97, x_center_norm: 0.5 },
                ],
                vision_result: {
                  wet_floor_or_spill: false,
                  detected_floor_number: 4,
                  scene_summary: "Corridor on Floor 4 in front of Room 443B door plaque.",
                },
              })
            }
            accessibilityRole="button"
            accessibilityLabel="Simulate arriving at Room 443B on Floor 4"
          >
            <Text style={styles.demoChipText}>4. Sim: Arrive 443B</Text>
          </Pressable>
        </ScrollView>
      </View>

      {/* GIANT BOTTOM-SCREEN PUSH-TO-TALK BUTTON */}
      <Pressable
        style={({ pressed }) => [
          styles.pttButton,
          isRecording || pressed ? styles.pttButtonActive : null,
        ]}
        onPressIn={startPushToTalk}
        onPressOut={stopPushToTalk}
        accessible={true}
        accessibilityRole="button"
        accessibilityLabel="Push to Talk Voice Command Button"
        accessibilityHint="Press and hold anywhere on the bottom of the screen to speak your destination or ask where you are. Release to send."
      >
        {isRecording ? (
          <View style={styles.pttInner}>
            <ActivityIndicator size="large" color="#000000" />
            <Text style={styles.pttMainTextActive}>LISTENING... RELEASE TO SEND</Text>
            <Text style={styles.pttSubTextActive}>
              Speak destination building & room (e.g., "Room 443B")
            </Text>
          </View>
        ) : (
          <View style={styles.pttInner}>
            <Text style={styles.pttMainText}>HOLD TO SPEAK</Text>
            <Text style={styles.pttSubText}>
              PUSH-TO-TALK • WHISPER VOICE NAVIGATION
            </Text>
          </View>
        )}
      </Pressable>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000000",
  },
  cameraContainer: {
    flex: 1,
    position: "relative",
    justifyContent: "space-between",
    paddingHorizontal: 14,
    paddingTop: Platform.OS === "android" ? 36 : 10,
    paddingBottom: 10,
  },
  cameraFallback: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "#080808",
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
  },
  cameraFallbackText: {
    color: "#FFD600",
    fontSize: 16,
    fontWeight: "900",
    textAlign: "center",
    letterSpacing: 1,
  },
  cameraScrim: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(0, 0, 0, 0.72)",
  },
  priorityBanner: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 12,
    borderWidth: 3,
  },
  bannerSafe: {
    backgroundColor: "#0A192F",
    borderColor: "#00E5FF",
  },
  bannerMedium: {
    backgroundColor: "#332200",
    borderColor: "#FFD600",
  },
  bannerHigh: {
    backgroundColor: "#4A000E",
    borderColor: "#FF1744",
  },
  bannerArrived: {
    backgroundColor: "#003319",
    borderColor: "#00E676",
  },
  priorityTitle: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "900",
    letterSpacing: 0.8,
  },
  latencyBadge: {
    color: "#FFD600",
    fontSize: 13,
    fontWeight: "800",
  },
  telemetryGrid: {
    gap: 8,
  },
  telemetryCard: {
    backgroundColor: "#0D0D0D",
    borderWidth: 2,
    borderColor: "#333333",
    borderRadius: 10,
    paddingVertical: 8,
    paddingHorizontal: 14,
  },
  telemetryLabel: {
    color: "#00E5FF",
    fontSize: 11,
    fontWeight: "800",
    letterSpacing: 1,
  },
  telemetryValue: {
    color: "#FFFFFF",
    fontSize: 18,
    fontWeight: "900",
    marginTop: 2,
  },
  telemetryRow: {
    flexDirection: "row",
    gap: 10,
  },
  telemetryPill: {
    flex: 1,
    backgroundColor: "#0D0D0D",
    borderRadius: 10,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderWidth: 2,
  },
  pillCyan: {
    borderColor: "#00E5FF",
  },
  pillYellow: {
    borderColor: "#FFD600",
  },
  pillWarning: {
    borderColor: "#FF9100",
    backgroundColor: "#261400",
  },
  pillLabel: {
    color: "#CCCCCC",
    fontSize: 11,
    fontWeight: "800",
  },
  pillValue: {
    color: "#FFD600",
    fontSize: 22,
    fontWeight: "900",
    marginTop: 2,
  },
  guidanceBox: {
    backgroundColor: "#000000",
    borderWidth: 3,
    borderColor: "#FFD600",
    borderRadius: 14,
    padding: 16,
    minHeight: 125,
    justifyContent: "center",
  },
  guidanceHeader: {
    color: "#00E5FF",
    fontSize: 12,
    fontWeight: "900",
    letterSpacing: 1.2,
    marginBottom: 6,
  },
  guidanceText: {
    color: "#FFFFFF",
    fontSize: 22,
    lineHeight: 30,
    fontWeight: "900",
  },
  detectionsStrip: {
    backgroundColor: "#111111",
    borderRadius: 10,
    padding: 10,
    borderLeftWidth: 5,
    borderLeftColor: "#00E5FF",
  },
  stripTitle: {
    color: "#FFD600",
    fontSize: 13,
    fontWeight: "900",
  },
  stripSub: {
    color: "#E0E0E0",
    fontSize: 12,
    fontWeight: "700",
    marginTop: 4,
  },
  demoStrip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingVertical: 4,
  },
  demoChip: {
    backgroundColor: "#1A1A1A",
    borderWidth: 2,
    borderColor: "#00E5FF",
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 12,
  },
  demoChipDanger: {
    borderColor: "#FF1744",
  },
  demoChipSuccess: {
    borderColor: "#00E676",
  },
  demoChipText: {
    color: "#FFFFFF",
    fontSize: 13,
    fontWeight: "800",
  },
  pttButton: {
    height: 175,
    backgroundColor: "#FFD600",
    borderTopWidth: 5,
    borderTopColor: "#FFFFFF",
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 20,
  },
  pttButtonActive: {
    backgroundColor: "#00E5FF",
  },
  pttInner: {
    alignItems: "center",
    justifyContent: "center",
  },
  pttMainText: {
    color: "#000000",
    fontSize: 34,
    fontWeight: "900",
    letterSpacing: 1.5,
  },
  pttSubText: {
    color: "#000000",
    fontSize: 14,
    fontWeight: "900",
    marginTop: 6,
    letterSpacing: 0.8,
  },
  pttMainTextActive: {
    color: "#000000",
    fontSize: 24,
    fontWeight: "900",
    marginTop: 8,
  },
  pttSubTextActive: {
    color: "#000000",
    fontSize: 14,
    fontWeight: "800",
    marginTop: 4,
  },
});

