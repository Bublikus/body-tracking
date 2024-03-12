'use client';

import {useEffect, useRef} from 'react';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import * as THREE from 'three';

// Define the connections based on human anatomy and keypoints order
const connections = [
  [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], // Head to eyes
  [7, 11], [8, 12], // Ears to shoulders
  [11, 13], [13, 15], [15, 17], [17, 19], [19, 21], // Left arm
  [12, 14], [14, 16], [16, 18], [18, 20], [20, 22], // Right arm
  [11, 23], [23, 25], [25, 27], [27, 29], [29, 31], // Left leg
  [12, 24], [24, 26], [26, 28], [28, 30], [30, 32], // Right leg
  [23, 33], [24, 33], // Hips to body center
];

export const Video = () => {
  const sceneRef = useRef();
  const sceneElRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);

  // Initialize Three.js scene
  function initThreeJS(webcamElement) {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, webcamElement.width / webcamElement.height, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true });
    renderer.setSize(webcamElement.width, webcamElement.height);
    sceneElRef.current.appendChild?.(renderer.domElement);

    // Adjust camera position
    camera.position.z = 2;

    sceneRef.current = scene;
    rendererRef.current = renderer;
    cameraRef.current = camera;
  }

  // Draw the pose using Three.js
  function drawPose(pose) {
    const scene = sceneRef.current;
    const renderer = rendererRef.current;
    const camera = cameraRef.current;

    // Clear the scene
    while (scene.children.length > 0) {
      scene.remove(scene.children[0]);
    }

    const pointScoreThreshold = 0.2;
    const connectionScoreThreshold = 0.5;
    const scaleFactor = 5;
    const zOffset = -5;

    // Adjust these values to change the dot sizes and line thickness
    const sphereRadius = 0.2; // Larger for bigger dots
    const tubeRadius = 0.2; // Larger for thicker lines

    const materialRed = new THREE.MeshBasicMaterial({ color: 0xff0000 });
    const materialGreen = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
    const materialOrange = new THREE.MeshBasicMaterial({ color: 0xffa500 });
    const geometry = new THREE.SphereGeometry(sphereRadius, 32, 32);
    const lineMaterial = new THREE.MeshBasicMaterial({ color: 0x0000ff });

    const points = pose.keypoints3D.filter((kp) => kp.score > pointScoreThreshold).map((kp, index) => {
      let material;
      if (index === 0 || index === 34) {
        material = materialRed;
      } else if ([1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 36].includes(index)) {
        material = materialGreen;
      } else {
        material = materialOrange;
      }

      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.x = kp.x * scaleFactor;
      sphere.position.y = -kp.y * scaleFactor;
      sphere.position.z = (kp.z * scaleFactor) + zOffset;
      scene.add(sphere);

      return { sphere, score: kp.score };
    });

    connections.forEach(([start, end]) => {
      if (points[start]?.score > connectionScoreThreshold && points[end]?.score > connectionScoreThreshold) {
        const startVec = new THREE.Vector3(points[start].sphere.position.x, points[start].sphere.position.y, points[start].sphere.position.z);
        const endVec = new THREE.Vector3(points[end].sphere.position.x, points[end].sphere.position.y, points[end].sphere.position.z);

        // Create a curve and then use TubeGeometry
        const path = new THREE.LineCurve3(startVec, endVec);
        const tubeGeometry = new THREE.TubeGeometry(path, 20, tubeRadius, 8, false);

        const line = new THREE.Mesh(tubeGeometry, lineMaterial);
        scene.add(line);
      }
    });

    renderer.render(scene, camera);
  }

  // Load the model
  async function loadPoseDetector() {
    const model = poseDetection.SupportedModels.BlazePose;
    const detectorConfig = {
      runtime: 'tfjs',
      enableSmoothing: true,
      modelType: 'full'
    };
    const detector = await poseDetection.createDetector(model, detectorConfig);
    return detector;
  }

  // Initialize the application
  async function initApp() {
    await tf.ready()

    const detector = await loadPoseDetector();
    const webcamElement = document.getElementById('webcam') as HTMLVideoElement;
    initThreeJS(webcamElement);

    const processWebcamFrame = async () => {
      const image = tf.browser.fromPixels(webcamElement);
      const poses = await detector.estimatePoses(image, {flipHorizontal: true});

      if (poses.length > 0) {
        drawPose(poses[0]);
      }

      image.dispose();
      requestAnimationFrame(processWebcamFrame);
    };

    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        webcamElement.srcObject = stream;
        webcamElement.onloadedmetadata = () => {
          webcamElement.play();
          requestAnimationFrame(processWebcamFrame);
        };
      });
  }

  useEffect(() => {
    initApp();
  }, []);

  return (
    <div className="relative">
      <video id="webcam" autoPlay playsInline width="640" height="480" className="-scale-x-100"></video>
      <div ref={sceneElRef} className="absolute inset-0 -scale-x-100"/>
    </div>
  );
}