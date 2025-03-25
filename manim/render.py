from manim import *

class CompressionIgnitionEngine(Scene):
    def construct(self):
        # Define engine components - shift everything to the right
        RIGHT_SHIFT = 3  # Amount to shift the engine to the right
        
        cylinder = Rectangle(height=5, width=3, color=GRAY, fill_opacity=0.3).move_to(RIGHT * RIGHT_SHIFT)
        piston = Rectangle(height=1, width=2.8, color=BLUE_D, fill_opacity=0.8).move_to(DOWN * 1.5 + RIGHT * RIGHT_SHIFT)
        intake_valve = Circle(radius=0.3, color=GREEN).move_to(UP * 2 + LEFT * 0.8 + RIGHT * RIGHT_SHIFT)
        exhaust_valve = Circle(radius=0.3, color=RED).move_to(UP * 2 + RIGHT * 0.8 + RIGHT * RIGHT_SHIFT)
        injector = Triangle(color=YELLOW, fill_opacity=0.8).scale(0.3).move_to(UP * 2 + RIGHT * RIGHT_SHIFT)
        
        # Create title and stroke label (now on the left)
        title = Text("4-Stroke Compression Ignition Engine", font_size=36).to_edge(UP)
        stroke_label = Text("", font_size=28).move_to(LEFT * 4)
        
        # Add name in bottom left with padding
        name = Text("Animated by: Ashish Prajapati", font_size=20, color=LIGHT_GRAY)
        name.to_corner(DOWN + LEFT, buff=0.5)  # buff parameter adds padding

        # Add components to the scene
        self.add(cylinder, piston, intake_valve, exhaust_valve, injector, title)
        self.add(stroke_label, name)
        
        # INTAKE STROKE
        stroke_label.become(Text("Intake Stroke", font_size=28).move_to(LEFT * 4))
        self.wait(0.5)
        
        # Open intake valve
        self.play(intake_valve.animate.shift(DOWN * 0.3), run_time=0.5)
        
        # Create air particles (blue dots)
        air_particles = VGroup(*[
            Dot(point=[np.random.uniform(-1, 1) + RIGHT_SHIFT, np.random.uniform(0, 2), 0], 
                radius=0.05, color=BLUE_A)
            for _ in range(20)
        ])
        self.play(FadeIn(air_particles))
        
        # Piston moves down (intake)
        self.play(
            piston.animate.shift(DOWN * 2),
            air_particles.animate.shift(DOWN * 1),
            run_time=1.5
        )
        
        # Close intake valve
        self.play(intake_valve.animate.shift(UP * 0.3), run_time=0.5)
        self.wait(0.5)
        
        # COMPRESSION STROKE
        stroke_label.become(Text("Compression Stroke", font_size=28).move_to(LEFT * 4))
        
        # Piston moves up (compression)
        self.play(
            piston.animate.shift(UP * 2),
            air_particles.animate.arrange_in_grid(rows=4, buff=0.2).scale(0.8).move_to(UP * 0.5 + RIGHT * RIGHT_SHIFT),
            run_time=1.5
        )
        self.wait(0.5)
        
        # POWER STROKE
        stroke_label.become(Text("Power Stroke", font_size=28).move_to(LEFT * 4))
        
        # Fuel injection
        fuel_particles = VGroup(*[
            Dot(point=injector.get_center() + DOWN * 0.3 + 
                np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0), 0]), 
                radius=0.05, color=YELLOW)
            for _ in range(10)
        ])
        
        self.play(
            injector.animate.scale(1.2),
            FadeIn(fuel_particles),
            run_time=0.5
        )
        self.play(injector.animate.scale(1/1.2), run_time=0.3)
        
        # Combustion effect
        combustion = VGroup(*[
            Dot(
                point=UP * np.random.uniform(-0.5, 1) + LEFT * np.random.uniform(-1, 1) + RIGHT * RIGHT_SHIFT,
                radius=np.random.uniform(0.1, 0.3),
                color=color_gradient([RED, YELLOW], 2)[np.random.randint(0, 2)]
            )
            for _ in range(15)
        ])
        self.play(FadeIn(combustion), run_time=0.3)
        
        # Piston moves down (power)
        self.play(
            piston.animate.shift(DOWN * 2),
            combustion.animate.shift(DOWN * 1.5).scale(1.5).set_opacity(0.5),
            air_particles.animate.shift(DOWN * 1.5),
            fuel_particles.animate.shift(DOWN * 1.5),
            run_time=1.5
        )
        
        self.play(
            FadeOut(combustion),
            FadeOut(fuel_particles),
            run_time=0.5
        )
        self.wait(0.5)
        
        # EXHAUST STROKE
        stroke_label.become(Text("Exhaust Stroke", font_size=28).move_to(LEFT * 4))
        
        # Open exhaust valve
        self.play(exhaust_valve.animate.shift(DOWN * 0.3), run_time=0.5)
        
        # Piston moves up (exhaust)
        exhaust_particles = air_particles.copy().set_color(GRAY)
        self.play(
            piston.animate.shift(UP * 2),
            exhaust_particles.animate.shift(UP * 3 + RIGHT * 0.8),
            FadeOut(air_particles),
            run_time=1.5
        )
        
        # Fade out exhaust
        self.play(
            FadeOut(exhaust_particles),
            run_time=0.5
        )
        
        # Close exhaust valve
        self.play(exhaust_valve.animate.shift(UP * 0.3), run_time=0.5)
        
        # Complete cycle message
        stroke_label.become(Text("Cycle Complete", font_size=28).move_to(LEFT * 4))
        self.wait(1)
