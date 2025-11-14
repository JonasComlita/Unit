// src/hooks/useGame.ts
import { useState, useEffect, useRef } from 'react';
import { useScene } from 'react-babylonjs';

// Mobile Camera Controller Component
const MobileCameraController: React.FC = () => {
    const scene = useScene();
    const [touches, setTouches] = useState<Map<number, { x: number; y: number }>>(new Map());
    const lastPinchDistance = useRef<number>(0);

    useEffect(() => {
        if (!scene || !scene.activeCamera) return;

        const camera = scene.activeCamera;
        
        const handlePointerDown = (evt: PointerEvent) => {
            setTouches(prev => {
                const newTouches = new Map(prev);
                newTouches.set(evt.pointerId, { x: evt.clientX, y: evt.clientY });
                return newTouches;
            });
        };

        const handlePointerMove = (evt: PointerEvent) => {
            setTouches(prev => {
                if (!prev.has(evt.pointerId)) return prev;
                
                const oldTouch = prev.get(evt.pointerId)!;
                const newTouches = new Map(prev);
                newTouches.set(evt.pointerId, { x: evt.clientX, y: evt.clientY });

                // Two-finger rotation
                if (newTouches.size === 2) {
                    const touchArray = Array.from(newTouches.values());
                    const dx = touchArray[0].x - touchArray[1].x;
                    const dy = touchArray[0].y - touchArray[1].y;
                    const distance = Math.sqrt(dx * dx + dy * dy);

                    // Pinch to zoom
                    if (lastPinchDistance.current > 0) {
                        const delta = distance - lastPinchDistance.current;
                        if ('radius' in camera) {
                            (camera as any).radius = Math.max(20, Math.min(80, (camera as any).radius - delta * 0.1));
                        }
                    }
                    lastPinchDistance.current = distance;

                    // Pan rotation
                    const oldArray = Array.from(prev.values());
                    if (oldArray.length === 2) {
                        const avgDx = ((touchArray[0].x - oldArray[0].x) + (touchArray[1].x - oldArray[1].x)) / 2;
                        if ('alpha' in camera) {
                            (camera as any).alpha += avgDx * 0.01;
                        }
                    }
                } else if (newTouches.size === 1) {
                    // Single finger rotation
                    const dx = evt.clientX - oldTouch.x;
                    if ('alpha' in camera) {
                        (camera as any).alpha += dx * 0.01;
                    }
                }

                return newTouches;
            });
        };

        const handlePointerUp = (evt: PointerEvent) => {
            setTouches(prev => {
                const newTouches = new Map(prev);
                newTouches.delete(evt.pointerId);
                if (newTouches.size < 2) {
                    lastPinchDistance.current = 0;
                }
                return newTouches;
            });
        };

        const canvas = scene.getEngine().getRenderingCanvas();
        if (canvas) {
            canvas.addEventListener('pointerdown', handlePointerDown);
            canvas.addEventListener('pointermove', handlePointerMove);
            canvas.addEventListener('pointerup', handlePointerUp);
            canvas.addEventListener('pointercancel', handlePointerUp);

            return () => {
                canvas.removeEventListener('pointerdown', handlePointerDown);
                canvas.removeEventListener('pointermove', handlePointerMove);
                canvas.removeEventListener('pointerup', handlePointerUp);
                canvas.removeEventListener('pointercancel', handlePointerUp);
            };
        }
    }, [scene]);

    return null;
};

export default MobileCameraController;
