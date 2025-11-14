// src/game/visualEffects.ts
// Complete visual enhancement system for Unit Strategy Game

import {
    Scene,
    Color3,
    Color4,
    Vector3,
    StandardMaterial,
    PBRMetallicRoughnessMaterial,
    GlowLayer,
    ParticleSystem,
    Texture,
    Animation,
    CubicEase,
    EasingFunction,
    Mesh,
    AbstractMesh,
    DirectionalLight,
    ShadowGenerator,
    DefaultRenderingPipeline,
    Sound,
    HemisphericLight
} from '@babylonjs/core';

export interface VisualEffectsConfig {
    enableParticles: boolean;
    enableGlow: boolean;
    enableAnimations: boolean;
    enableShadows: boolean;
    enablePostProcessing: boolean;
    quality: 'low' | 'medium' | 'high';
}

export class VisualEffectsManager {
    private scene: Scene;
    private config: VisualEffectsConfig;
    private glowLayer?: GlowLayer;
    private shadowGenerator?: ShadowGenerator;
    private activeAnimations: Map<string, Animation> = new Map();
    private activeSounds: Map<string, Sound> = new Map();
    private isMobile: boolean;

    constructor(scene: Scene, config?: Partial<VisualEffectsConfig>) {
        this.scene = scene;
        this.isMobile = this.detectMobile();
        
        // Default config based on device
        this.config = {
            enableParticles: true,
            enableGlow: true,
            enableAnimations: true,
            enableShadows: !this.isMobile,
            enablePostProcessing: !this.isMobile,
            quality: this.isMobile ? 'medium' : 'high',
            ...config
        };

        this.initialize();
    }

    private initialize() {
        // Set up glow layer
        if (this.config.enableGlow) {
            this.setupGlowLayer();
        }

        // Set up improved lighting
        this.setupLighting();

        // Set up shadows (desktop only)
        if (this.config.enableShadows) {
            this.setupShadows();
        }

        // Set up post-processing (desktop only)
        if (this.config.enablePostProcessing) {
            this.setupPostProcessing();
        }
    }

    // ==========================================
    // MATERIALS
    // ==========================================

    /**
     * Create enhanced piece material with PBR
     */
    createPieceMaterial(
        playerId: string,
        isSelected: boolean = false,
        isHighlighted: boolean = false
    ): PBRMetallicRoughnessMaterial {
        const material = new PBRMetallicRoughnessMaterial(
            `piece-${playerId}-${Date.now()}`,
            this.scene
        );

        // Player colors
        const colors = {
            Player1: new Color3(0.29, 0.56, 0.89), // Blue
            Player2: new Color3(0.82, 0.13, 0.11)  // Red
        };

        material.baseColor = colors[playerId as keyof typeof colors];
        material.metallic = 0.7;
        material.roughness = 0.2;

        // Add subtle environment reflection
        // BabylonJS PBRMetallicRoughnessMaterial does not have environmentIntensity, use environmentTextureLevel if needed
        // Example: material.environmentTextureLevel = 0.5;

        // Emissive glow when selected or highlighted
        if (isSelected) {
            material.emissiveColor = material.baseColor.scale(0.4);
            // BabylonJS PBRMetallicRoughnessMaterial does not have emissiveIntensity, use emissiveColor only
            if (this.config.enableGlow) {
                this.glowLayer?.addIncludedOnlyMesh(material as any);
            }
        } else if (isHighlighted) {
            material.emissiveColor = material.baseColor.scale(0.2);
        }

        return material;
    }

    /**
     * Create force field material with fresnel effect
     */
    createForceFieldMaterial(
        playerId: string,
        force: number
    ): PBRMetallicRoughnessMaterial {
        const material = new PBRMetallicRoughnessMaterial(
            `force-${playerId}-${Date.now()}`,
            this.scene
        );

        const colors = {
            Player1: new Color3(0.29, 0.56, 0.89),
            Player2: new Color3(0.82, 0.13, 0.11)
        };

        const baseColor = colors[playerId as keyof typeof colors];
        material.baseColor = baseColor;
        
        // Alpha based on force (stronger = more visible)
        material.alpha = 0.1 + (force / 10) * 0.15;
        material.metallic = 0.1;
        material.roughness = 0.9;

        // Fresnel for edge glow (energy field effect)
    material.emissiveColor = baseColor.scale(0.6);
    // BabylonJS PBRMetallicRoughnessMaterial does not have emissiveIntensity, use emissiveColor only

        // Enable alpha blending
        material.transparencyMode = 2; // ALPHABLEND

        return material;
    }

    /**
     * Create vertex indicator material with pulsing
     */
    createVertexMaterial(
        actionType?: 'placement' | 'infusion' | 'movement' | 'attack' | 'pincer'
    ): StandardMaterial {
        const material = new StandardMaterial(
            `vertex-${actionType || 'default'}-${Date.now()}`,
            this.scene
        );

        material.diffuseColor = new Color3(0.7, 0.7, 0.9);
        material.specularColor = new Color3(0.2, 0.2, 0.2);

        // Action-specific colors
        const actionColors = {
            placement: new Color3(0.31, 0.78, 0.47),  // Green
            infusion: new Color3(1, 0.75, 0),         // Yellow
            movement: new Color3(0, 0.81, 0.82),      // Cyan
            attack: new Color3(0.86, 0.08, 0.24),     // Red
            pincer: new Color3(0.55, 0, 1)            // Purple
        };

        if (actionType && actionColors[actionType]) {
            material.emissiveColor = actionColors[actionType];
        }

        return material;
    }

    /**
     * Create indicator ring material
     */
    createIndicatorRingMaterial(color: Color3): StandardMaterial {
        const material = new StandardMaterial(`ring-${Date.now()}`, this.scene);
        material.emissiveColor = color;
        material.alpha = 0.8;
        material.disableLighting = true;
        return material;
    }

    // ==========================================
    // PARTICLES
    // ==========================================

    /**
     * Create particle effect for placement
     */
    createPlacementParticles(position: Vector3) {
        if (!this.config.enableParticles) return;

        const particles = new ParticleSystem('placement', 30, this.scene);
        
        // Use simple procedural texture (no file loading)
        particles.particleTexture = this.createParticleTexture();
        
        particles.emitter = position;
        particles.minEmitBox = new Vector3(-0.2, 0, -0.2);
        particles.maxEmitBox = new Vector3(0.2, 0.2, 0.2);
        
        particles.color1 = new Color4(0.31, 0.78, 0.47, 1); // Green
        particles.color2 = new Color4(0.2, 0.9, 0.6, 1);
        particles.colorDead = new Color4(1, 1, 1, 0);
        
        particles.minSize = 0.05;
        particles.maxSize = 0.15;
        particles.minLifeTime = 0.2;
        particles.maxLifeTime = 0.5;
        
        particles.emitRate = 100;
        particles.blendMode = ParticleSystem.BLENDMODE_ADD;
        
        particles.gravity = new Vector3(0, -1, 0);
        
        particles.direction1 = new Vector3(-0.5, 1, -0.5);
        particles.direction2 = new Vector3(0.5, 2, 0.5);
        
        particles.minAngularSpeed = 0;
        particles.maxAngularSpeed = Math.PI;
        
        particles.minEmitPower = 1;
        particles.maxEmitPower = 2;
        particles.updateSpeed = 0.01;
        
        particles.start();
        
        // Auto cleanup
        setTimeout(() => {
            particles.stop();
            setTimeout(() => particles.dispose(), 500);
        }, 300);
    }

    /**
     * Create particle effect for energy infusion
     */
    createInfusionParticles(position: Vector3) {
        if (!this.config.enableParticles) return;

        const particles = new ParticleSystem('infusion', 50, this.scene);
        particles.particleTexture = this.createParticleTexture();
        
        particles.emitter = position;
        particles.minEmitBox = new Vector3(-0.3, -0.3, -0.3);
        particles.maxEmitBox = new Vector3(0.3, -0.1, 0.3);
        
        particles.color1 = new Color4(1, 0.9, 0, 1); // Yellow
        particles.color2 = new Color4(1, 0.6, 0, 1); // Orange
        particles.colorDead = new Color4(1, 1, 1, 0);
        
        particles.minSize = 0.03;
        particles.maxSize = 0.1;
        particles.minLifeTime = 0.3;
        particles.maxLifeTime = 0.8;
        
        particles.emitRate = 80;
        particles.blendMode = ParticleSystem.BLENDMODE_ADD;
        
        // Particles rise upward (energy flowing in)
        particles.direction1 = new Vector3(-0.2, 0.5, -0.2);
        particles.direction2 = new Vector3(0.2, 1.5, 0.2);
        
        particles.minEmitPower = 0.5;
        particles.maxEmitPower = 1.5;
        particles.updateSpeed = 0.01;
        
        particles.start();
        
        setTimeout(() => {
            particles.stop();
            setTimeout(() => particles.dispose(), 800);
        }, 500);
    }

    /**
     * Create particle effect for attack clash
     */
    createAttackParticles(position: Vector3, attackerColor: Color3, defenderColor: Color3) {
        if (!this.config.enableParticles) return;

        // Attacker particles
        const attackerParticles = new ParticleSystem('attack', 40, this.scene);
        attackerParticles.particleTexture = this.createParticleTexture();
        
        attackerParticles.emitter = position;
        attackerParticles.minEmitBox = new Vector3(-0.3, -0.3, -0.3);
        attackerParticles.maxEmitBox = new Vector3(0.3, 0.3, 0.3);
        
        const c1 = attackerColor;
        attackerParticles.color1 = new Color4(c1.r, c1.g, c1.b, 1);
        attackerParticles.color2 = new Color4(1, 1, 1, 1);
        attackerParticles.colorDead = new Color4(0.5, 0.5, 0.5, 0);
        
        attackerParticles.minSize = 0.1;
        attackerParticles.maxSize = 0.25;
        attackerParticles.minLifeTime = 0.2;
        attackerParticles.maxLifeTime = 0.6;
        
        attackerParticles.emitRate = 150;
        attackerParticles.blendMode = ParticleSystem.BLENDMODE_ADD;
        
        attackerParticles.direction1 = new Vector3(-2, -2, -2);
        attackerParticles.direction2 = new Vector3(2, 2, 2);
        
        attackerParticles.minEmitPower = 2;
        attackerParticles.maxEmitPower = 4;
        attackerParticles.updateSpeed = 0.01;
        
        attackerParticles.start();
        
        setTimeout(() => {
            attackerParticles.stop();
            setTimeout(() => attackerParticles.dispose(), 600);
        }, 200);
    }

    /**
     * Create victory celebration particles
     */
    createVictoryParticles(position: Vector3) {
        if (!this.config.enableParticles) return;

        const particles = new ParticleSystem('victory', 200, this.scene);
        particles.particleTexture = this.createParticleTexture();
        
        particles.emitter = position;
        particles.minEmitBox = new Vector3(-2, 0, -2);
        particles.maxEmitBox = new Vector3(2, 0, 2);
        
        particles.color1 = new Color4(1, 0.84, 0, 1); // Gold
        particles.color2 = new Color4(1, 0.95, 0.3, 1);
        particles.colorDead = new Color4(1, 1, 1, 0);
        
        particles.minSize = 0.1;
        particles.maxSize = 0.3;
        particles.minLifeTime = 1.0;
        particles.maxLifeTime = 2.0;
        
        particles.emitRate = 100;
        particles.blendMode = ParticleSystem.BLENDMODE_ADD;
        
        particles.gravity = new Vector3(0, -2, 0);
        particles.direction1 = new Vector3(-3, 5, -3);
        particles.direction2 = new Vector3(3, 8, 3);
        
        particles.minEmitPower = 3;
        particles.maxEmitPower = 6;
        particles.updateSpeed = 0.01;
        
        particles.start();
        
        setTimeout(() => {
            particles.stop();
            setTimeout(() => particles.dispose(), 2000);
        }, 1500);
    }

    // ==========================================
    // ANIMATIONS
    // ==========================================

    /**
     * Animate piece placement (materialize effect)
     */
    animatePlacement(mesh: AbstractMesh, onComplete?: () => void) {
        if (!this.config.enableAnimations) {
            if (onComplete) onComplete();
            return;
        }

        // Start small and scale up
        mesh.scaling = new Vector3(0.1, 0.1, 0.1);
        
        const scaleAnimation = new Animation(
            'placementScale',
            'scaling',
            60,
            Animation.ANIMATIONTYPE_VECTOR3,
            Animation.ANIMATIONLOOPMODE_CONSTANT
        );

        const keys = [
            { frame: 0, value: new Vector3(0.1, 0.1, 0.1) },
            { frame: 15, value: new Vector3(1.2, 1.2, 1.2) },
            { frame: 30, value: new Vector3(1, 1, 1) }
        ];

        scaleAnimation.setKeys(keys);

        const easingFunction = new CubicEase();
        easingFunction.setEasingMode(EasingFunction.EASINGMODE_EASEOUT);
        scaleAnimation.setEasingFunction(easingFunction);

        mesh.animations = [scaleAnimation];
        
        this.scene.beginAnimation(mesh, 0, 30, false, 1, onComplete);
        
        // Add particles
        this.createPlacementParticles(mesh.position);
    }

    /**
     * Animate piece movement along path
     */
    animateMovement(
        mesh: AbstractMesh,
        fromPosition: Vector3,
        toPosition: Vector3,
        onComplete?: () => void
    ) {
        if (!this.config.enableAnimations) {
            mesh.position = toPosition;
            if (onComplete) onComplete();
            return;
        }

        // Create arc path (Bezier curve)
        const midPoint = Vector3.Lerp(fromPosition, toPosition, 0.5);
        const height = Vector3.Distance(fromPosition, toPosition) * 0.3;
        midPoint.y += height;

        const posAnimation = new Animation(
            'movement',
            'position',
            60,
            Animation.ANIMATIONTYPE_VECTOR3,
            Animation.ANIMATIONLOOPMODE_CONSTANT
        );

        const keys = [
            { frame: 0, value: fromPosition },
            { frame: 15, value: midPoint },
            { frame: 30, value: toPosition }
        ];

        posAnimation.setKeys(keys);

        const easingFunction = new CubicEase();
        easingFunction.setEasingMode(EasingFunction.EASINGMODE_EASEINOUT);
        posAnimation.setEasingFunction(easingFunction);

        mesh.animations = [posAnimation];
        
        this.scene.beginAnimation(mesh, 0, 30, false, 1, onComplete);
    }

    /**
     * Pulse animation for selected/highlighted objects
     */
    animatePulse(mesh: AbstractMesh, duration: number = 2000) {
        if (!this.config.enableAnimations) return;

        const scaleAnimation = new Animation(
            'pulse',
            'scaling',
            60,
            Animation.ANIMATIONTYPE_VECTOR3,
            Animation.ANIMATIONLOOPMODE_CYCLE
        );

        const baseScale = mesh.scaling.clone();
        const keys = [
            { frame: 0, value: baseScale },
            { frame: 30, value: baseScale.scale(1.1) },
            { frame: 60, value: baseScale }
        ];

        scaleAnimation.setKeys(keys);
        mesh.animations = [scaleAnimation];
        
        const animatable = this.scene.beginAnimation(mesh, 0, 60, true, 1);
        
        // Store for cleanup
        this.activeAnimations.set(mesh.uniqueId.toString(), scaleAnimation);
        
        // Auto-stop after duration
        setTimeout(() => {
            animatable.stop();
            this.activeAnimations.delete(mesh.uniqueId.toString());
        }, duration);
    }

    /**
     * Shake effect for attacks
     */
    animateCameraShake(intensity: number = 0.05, duration: number = 300) {
        if (!this.config.enableAnimations || !this.scene.activeCamera) return;

        const camera = this.scene.activeCamera;
        const originalPosition = camera.position.clone();
        const startTime = Date.now();

        const shake = () => {
            const elapsed = Date.now() - startTime;
            if (elapsed > duration) {
                camera.position = originalPosition;
                return;
            }

            const progress = elapsed / duration;
            const currentIntensity = intensity * (1 - progress);

            camera.position.x = originalPosition.x + (Math.random() - 0.5) * currentIntensity;
            camera.position.y = originalPosition.y + (Math.random() - 0.5) * currentIntensity;
            camera.position.z = originalPosition.z + (Math.random() - 0.5) * currentIntensity;

            requestAnimationFrame(shake);
        };

        shake();
    }

    // ==========================================
    // SETUP METHODS
    // ==========================================

    private setupGlowLayer() {
        const textureSize = this.config.quality === 'high' ? 1024 : 512;
        
        this.glowLayer = new GlowLayer('glow', this.scene, {
            mainTextureFixedSize: textureSize,
            blurKernelSize: this.config.quality === 'high' ? 32 : 16
        });
        
        this.glowLayer.intensity = 0.6;
    }

    private setupLighting() {
        // Main directional light for depth
        const dirLight = new DirectionalLight(
            'mainLight',
            new Vector3(-1, -2, -1),
            this.scene
        );
        dirLight.intensity = 0.8;
        dirLight.specular = new Color3(0.2, 0.2, 0.2);
        
        // Keep existing hemisphere lights but adjust
        const hemiLight1 = new HemisphericLight(
            'hemi1',
            new Vector3(0, 1, 0),
            this.scene
        );
        hemiLight1.intensity = 0.5;
        hemiLight1.groundColor = new Color3(0.2, 0.2, 0.3);
        
        const hemiLight2 = new HemisphericLight(
            'hemi2',
            new Vector3(0, -1, 0),
            this.scene
        );
        hemiLight2.intensity = 0.3;
    }

    private setupShadows() {
        const light = this.scene.lights.find(l => l instanceof DirectionalLight) as DirectionalLight;
        if (!light) return;

        this.shadowGenerator = new ShadowGenerator(1024, light);
        this.shadowGenerator.useBlurExponentialShadowMap = true;
        this.shadowGenerator.blurScale = 2;
        this.shadowGenerator.setDarkness(0.3);
    }

    private setupPostProcessing() {
        if (!this.scene.activeCamera) return;

        const pipeline = new DefaultRenderingPipeline(
            'default',
            true,
            this.scene,
            [this.scene.activeCamera]
        );

        // Subtle bloom
        pipeline.bloomEnabled = true;
        pipeline.bloomThreshold = 0.8;
        pipeline.bloomWeight = 0.25;
        pipeline.bloomKernel = 32;

        // Slight sharpening
        pipeline.sharpenEnabled = true;
        pipeline.sharpen.edgeAmount = 0.15;
        pipeline.sharpen.colorAmount = 0.1;

        // Subtle chromatic aberration for style
        pipeline.chromaticAberrationEnabled = true;
        pipeline.chromaticAberration.aberrationAmount = 5;
    }

    // ==========================================
    // UTILITIES
    // ==========================================

    private createParticleTexture(): Texture {
        // Create procedural particle texture (no file loading needed)
        const size = 64;
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d')!;

        // Create radial gradient
        const gradient = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
        gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
        gradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.5)');
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, size, size);

        const texture = new Texture(canvas.toDataURL(), this.scene);
        return texture;
    }

    private detectMobile(): boolean {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
            navigator.userAgent
        );
    }

    /**
     * Add mesh to shadow casters
     */
    addShadowCaster(mesh: AbstractMesh) {
        if (this.shadowGenerator && this.config.enableShadows) {
            this.shadowGenerator.addShadowCaster(mesh);
        }
    }

    /**
     * Enable shadow receiving on mesh
     */
    enableShadowReceiver(mesh: AbstractMesh) {
        if (this.config.enableShadows) {
            mesh.receiveShadows = true;
        }
    }

    /**
     * Clean up all resources
     */
    dispose() {
        this.glowLayer?.dispose();
        this.shadowGenerator?.dispose();
        this.activeAnimations.clear();
        this.activeSounds.forEach(sound => sound.dispose());
        this.activeSounds.clear();
    }

    /**
     * Update quality settings on the fly
     */
    updateQuality(quality: 'low' | 'medium' | 'high') {
        this.config.quality = quality;
        
        if (quality === 'low') {
            this.config.enableParticles = false;
            this.config.enableShadows = false;
            this.config.enablePostProcessing = false;
        } else if (quality === 'medium') {
            this.config.enableParticles = true;
            this.config.enableShadows = false;
            this.config.enablePostProcessing = false;
        } else {
            this.config.enableParticles = true;
            this.config.enableShadows = !this.isMobile;
            this.config.enablePostProcessing = !this.isMobile;
        }
    }
}

export default VisualEffectsManager;