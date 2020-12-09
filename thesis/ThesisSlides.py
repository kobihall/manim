#!/usr/bin/env python

from manimlib.imports import *


class HeaderMobject(TextMobject):
    CONFIG = {
        "scale_factor": 1,
        "color": BLUE,
        "bufffact": 1.5
    }
    def __init__(self,*text_parts,**kwargs):
        TextMobject.__init__(self, *text_parts, **kwargs)
        self.scale(self.scale_factor)
        self.to_corner(UL,buff=self.bufffact)
        self.set_color(self.color)

class Floor(Line):
    CONFIG = {
        "tick_spacing": 0.5,
        "tick_length": 0.25,
        "tick_style": {
            "stroke_width": 1,
            "stroke_color": WHITE,
        },
    }
    def __init__(self, height, **kwargs):
        Line.__init__(self, FRAME_WIDTH * LEFT / 2+height*UP, FRAME_WIDTH * RIGHT / 2+height*UP, **kwargs)
        self.height = height
        self.ticks = self.get_ticks()
        self.add(self.ticks)

    def get_ticks(self):
        n_lines = int(FRAME_WIDTH / self.tick_spacing)
        lines = VGroup(*[
            Line(ORIGIN, self.tick_length * UR).shift(n * self.tick_spacing * RIGHT)
            for n in range(n_lines)
        ])
        lines.set_style(**self.tick_style)
        lines.move_to(self, UR)
        return lines

class Box(Mobject):
    CONFIG = {
        "tick_spacing": 0.5,
        "tick_length": 0.25,
        "tick_style": {
            "stroke_width": 1,
            "stroke_color": WHITE,
        },
    }
    def __init__(self, height, width, **kwargs):
        Mobject.__init__(self, **kwargs)
        self.TPLF = width * LEFT / 2+height*UP/2
        self.TPRT = width * RIGHT / 2+height*UP/2
        self.BTLF = width * LEFT / 2+height*DOWN/2
        self.BTRT = width * RIGHT / 2+height*DOWN/2
        self.topline = Line(self.TPRT, self.TPLF, **kwargs)
        self.botline = Line(self.BTLF, self.BTRT, **kwargs)
        self.rightline = Line(self.BTRT,self.TPRT, **kwargs)
        self.leftline = Line(self.TPLF, self.BTLF, **kwargs)
        self.height = height
        self.width = width
        self.topticks = self.get_ticks(self.TPLF,self.TPRT)
        self.botticks = self.get_ticks(self.BTRT,self.BTLF)
        self.rightticks = self.get_ticks(self.TPRT,self.BTRT)
        self.leftticks = self.get_ticks(self.BTLF, self.TPLF)
        self.add(self.topline, self.botline, self.rightline, self.leftline,self.topticks, self.botticks, self.leftticks,self.rightticks)

    def get_ticks(self, start, end):
        n_lines = int(np.linalg.norm(end - start)/ self.tick_spacing)
        lines = VGroup(*[
            Line(ORIGIN, self.tick_length * UR).shift(n * self.tick_spacing * (start - end)/np.linalg.norm(end - start))
            for n in range(n_lines)
        ])
        lines.set_style(**self.tick_style)
        lines.move_to((start+end)/2, -(start+end)/np.linalg.norm(end + start) )
        return lines

class GaussianDistributionWrapper(Line):
    """
    This is meant to encode a 2d normal distribution as
    a mobject (so as to be able to have it be interpolated
    during animations).  It is a line whose center is the mean
    mu of a distribution, and whose radial vector (center to end)
    is the distribution's standard deviation
    """
    CONFIG = {
        "stroke_width" : 0,
        "mu" : ORIGIN,
        "sigma" : RIGHT,
    }
    def __init__(self, **kwargs):
        Line.__init__(self, ORIGIN, RIGHT, **kwargs)
        self.change_parameters(self.mu, self.sigma)

    def change_parameters(self, mu = None, sigma = None):
        curr_mu, curr_sigma = self.get_parameters()
        mu = mu if mu is not None else curr_mu
        sigma = sigma if sigma is not None else curr_sigma
        self.put_start_and_end_on(mu - sigma, mu + sigma)
        return self

    def get_parameters(self):
        """ Return mu_x, mu_y, sigma_x, sigma_y"""
        center, end = self.get_center(), self.get_end()
        return center, end-center

    def get_random_points(self, size = 1):
        mu, sigma = self.get_parameters()
        return np.array([
            np.array([
                np.random.normal(mu_coord, sigma_coord)
                for mu_coord, sigma_coord in zip(mu, sigma)
            ])
            for x in range(size)
        ])

class ProbabilityCloud(Mobject):
    CONFIG = {
        "fill_opacity" : 0.15,
        "n_copies" : 200,
        "gaussian_distribution_wrapper_config" : {},
    }
    def __init__(self, color = BLUE, hbound = 1, vbound = 1, **kwargs):
        Mobject.__init__(self, **kwargs)
        digest_config(self, kwargs)
        point = Dot()
        self.hbound = hbound
        self.vbound = vbound
        if "mu" not in self.gaussian_distribution_wrapper_config:
            self.gaussian_distribution_wrapper_config["mu"] = point.get_center()
        self.gaussian_distribution_wrapper = GaussianDistributionWrapper(
            **self.gaussian_distribution_wrapper_config
        )
        self.group = VGroup(*[
            point.copy().set_fill(opacity = self.fill_opacity, color = color)
            for x in range(self.n_copies)
        ])
        self.add(self.group)
        self.add_updater(lambda v: v.update_points())

    def update_points(self):
        points = self.gaussian_distribution_wrapper.get_random_points(len(self.group))
        for mob, point in zip(self.group, points):
            if np.abs(point[0]) < self.hbound and np.abs(point[1]) < self.vbound:
                mob.move_to(point)
        return self

# To watch one of these scenes, run the following:
# python -m manim example_scenes.py SquareToCircle -pl
#
# Use the flat -l for a faster rendering at a lower
# quality.
# Use -s to skip to the end and just save the final frame
# Use the -p to have the animation (or image, if -s was
# used) pop up once done.
# Use -n <number> to skip ahead to the n'th animation of a scene.
# Use -r <number> to specify a resolution (for example, -r 1080
# for a 1920x1080 video)

class ATitleSlide(Scene):
    def construct(self):
        Title = TextMobject(
                "The Nonlinear Schrödinger Equation",
                tex_to_color_map={"text": YELLOW}
            )
        Name = TextMobject(
                "An Honors Thesis by Kobi Hall",
                tex_to_color_map={"text": BLUE})
        Accessibility = TextMobject(
            "Puget Sound is committed to being accessible to all people. If you have questions about event accessibility,\
             please contact 253.879.3931, accessibility@pugetsound.edu or pugetsound.edu/accessibility")
        Date = TextMobject(
            "Wednesday, March 11th, 4PM")
        Location = TextMobject(
            "Thompson 193")
        Title.scale(1.5)
        group = VGroup(Title, Name)
        group.arrange_submobjects(DOWN)
        group.to_edge(UP, buff=1)
        Accessibility.scale(.4)
        Accessibility.to_corner(DL)
        Location.to_corner(DR)
        Date.to_corner(DR)
        Date.shift(UP)


        t_tracker = ValueTracker(0)
        w1 = self.get_wave(tr=t_tracker,k0=10,x0=-11, scale = 1.5)
        w2 = self.get_wave(tr=t_tracker,k0=-10,x0=11,t0=3, scale = 1.5)

        self.add(Title, Name)
        self.add(w1)
        #self.add(w1, Accessibility, Date, Location)
        #self.wait(1)
        self.play(
            t_tracker.set_value,3,
            rate_func=linear,
            run_time=3
            )
        self.add(w2)
        self.remove(w1)
        self.wait(2)
        self.play(
            t_tracker.set_value,t_tracker.get_value()+3,
            rate_func=linear,
            run_time=3
            )


    def get_wave(self,tr=None,**kwargs):
        def get_gaussian_func(x,c="r",t=0,k0=1,x0=0,t0=0, scale = 1):
            wave_part = np.exp(
                complex(0, ((k0+2*(t-t0)*(x-x0)/scale)*(x-x0)/scale-k0**2*(t-t0)/2)/(1+4*(t-t0)**2))
            )
            bell_part = np.exp(-((x-x0)/scale-k0*(t-t0))**2/(1+4*(t-t0)**2))
            amplitude = scale #(2/PI)**1/4 / (1+2j*(t-t0))**1/2
            if c == "r":
                return (amplitude*wave_part*bell_part).real
            if c == "i":
                return (amplitude*wave_part*bell_part).imag
        
        graphre = FunctionGraph(lambda x: get_gaussian_func(x))
        graphre.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="r",t=tr.get_value(),**kwargs)))
        
        graphim = FunctionGraph(lambda x: get_gaussian_func(x))
        graphim.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="i",t=tr.get_value(),**kwargs), color=BLUE))
        return VGroup(graphre,graphim)

class BBNLSEPreview(Scene):
    def construct(self):
        NLSE = TexMobject(
            r"i\hbar\pderiv{\Psi}{t}+\frac{\hbar^2}{2m}\pderiv{^2\Psi}{x^2}-\kappa\left|\Psi\right|^2\Psi=0"
            )
        Scary = TextMobject("Scary!")
        Scary.shift(2*DOWN)
        Scary.set_color(color=RED)

        self.play(Write(NLSE))

class BBNLSEPreview(Scene):
    def construct(self):
        NLSE = TexMobject(
            r"i\hbar\pderiv{\Psi}{t}+\frac{\hbar^2}{2m}\pderiv{^2\Psi}{x^2}-\kappa\left|\Psi\right|^2\Psi=0"
            )
        Scary = TextMobject("Scary!")
        Scary.shift(2*DOWN)
        Scary.set_color(color=RED)

        self.add(NLSE)
        self.play(Write(Scary))

class CAOutline(Scene):
    def construct(self):
        #Header = TextMobject("Outline")
        Header = HeaderMobject("Outline")
        bpt1 = TextMobject("-What is quantum mechanics?")
        bpt2 = TextMobject("-Differential Equations")
        bpt3 = TextMobject("-The linear Schrödinger equation")
        bpt4 = TextMobject("-The nonlinear Schrödinger equation")

        bpts = VGroup(bpt1,bpt2,bpt3,bpt4)
        bpts.arrange_submobjects(DOWN)
        bpt2.align_to(bpt1,LEFT)
        bpt3.align_to(bpt1,LEFT)
        bpt4.align_to(bpt1,LEFT)
        self.add(Header)
        self.play(Write(bpt1))
        #self.wait(1)
        #self.play(Write(bpt2))
        #self.wait(1)
        #self.play(Write(bpt3))
        #self.wait(1)
        #self.play(Write(bpt4))
        #self.wait(1)

class CBOutline(Scene):
    def construct(self):
        #Header = TextMobject("Outline")
        Header = HeaderMobject("Outline")
        bpt1 = TextMobject("-What is quantum mechanics?")
        bpt2 = TextMobject("-Differential equations")
        bpt3 = TextMobject("-The linear Schrödinger equation")
        bpt4 = TextMobject("-The nonlinear Schrödinger equation")

        bpts = VGroup(bpt1,bpt2,bpt3,bpt4)
        bpts.arrange_submobjects(DOWN)
        bpt2.align_to(bpt1,LEFT)
        bpt3.align_to(bpt1,LEFT)
        bpt4.align_to(bpt1,LEFT)
        self.add(Header, bpt1)
        self.play(Write(bpt2))

class CCOutline(Scene):
    def construct(self):
        #Header = TextMobject("Outline")
        Header = HeaderMobject("Outline")
        bpt1 = TextMobject("-What is quantum mechanics?")
        bpt2 = TextMobject("-Differential equations")
        bpt3 = TextMobject("-The linear Schrödinger equation")
        bpt4 = TextMobject("-The nonlinear Schrödinger equation")

        bpts = VGroup(bpt1,bpt2,bpt3,bpt4)
        bpts.arrange_submobjects(DOWN)
        bpt2.align_to(bpt1,LEFT)
        bpt3.align_to(bpt1,LEFT)
        bpt4.align_to(bpt1,LEFT)
        self.add(Header, bpt1, bpt2)
        self.play(Write(bpt3))

class CDOutline(Scene):
    def construct(self):
        #Header = TextMobject("Outline")
        Header = HeaderMobject("Outline")
        bpt1 = TextMobject("-What is quantum mechanics?")
        bpt2 = TextMobject("-Differential equations")
        bpt3 = TextMobject("-The linear Schrödinger equation")
        bpt4 = TextMobject("-The nonlinear Schrödinger equation")

        bpts = VGroup(bpt1,bpt2,bpt3,bpt4)
        bpts.arrange_submobjects(DOWN)
        bpt2.align_to(bpt1,LEFT)
        bpt3.align_to(bpt1,LEFT)
        bpt4.align_to(bpt1,LEFT)
        self.add(Header, bpt1, bpt2, bpt3)
        self.play(Write(bpt4))

class DAWhatIsQM(Scene):
    def construct(self):
        Header = HeaderMobject("What Is Quantum Mechanics?")
        ClassicalMech = TextMobject("classical mechanics")
        NotClassicalMech = TexMobject(r"\xcancel{\qquad \qquad \qquad \qquad}")
        NotClassicalMech.set_color(color=RED)
        #Header.scale(1)

        self.play(Write(Header))

class DBWhatIsQM(Scene):
    def construct(self):
        Header = HeaderMobject("What Is Quantum Mechanics?")
        ClassicalMech = TextMobject("classical mechanics")
        NotClassicalMech = TexMobject(r"\xcancel{\qquad \qquad \qquad \qquad}")
        NotClassicalMech.set_color(color=RED)
        #Header.scale(1)

        self.add(Header)
        self.play(Write(ClassicalMech))

class DCWhatIsQM(Scene):
    def construct(self):
        Header = HeaderMobject("What Is Quantum Mechanics?")
        ClassicalMech = TextMobject("classical mechanics")
        NotClassicalMech = TexMobject(r"\xcancel{\qquad \qquad \qquad \qquad}")
        NotClassicalMech.set_color(color=RED)
        #Header.scale(1)

        self.add(Header,ClassicalMech)
        self.play(Write(NotClassicalMech))

class EClassicalMechanics(Scene):

    def construct(self):
        floor = Floor(-3)

        dot = Dot(np.array([-5,-2,0]))
        dot.set_color(color=YELLOW_C)
        dot.scale(2)

        t_tracker = ValueTracker(PI/4)
        v0 = Arrow(dot.get_center(),dot.get_center()+UP)
        v0.add_updater(lambda v: v.put_start_and_end_on(start=dot.get_center(),end=dot.get_center()+np.array([np.cos(t_tracker.get_value()),np.sin(t_tracker.get_value()),0])))

        traj = ParametricFunction(lambda t: np.array([t, t, 0]))
        traj.add_updater(lambda v: v.__init__(lambda t: np.array([3*np.cos(t_tracker.get_value())*t, 3*np.sin(t_tracker.get_value())*t-9.8*t**2/2, 0])+dot.get_center(), \
         t_max = (3*np.sin(t_tracker.get_value()) + ( (3*np.sin(t_tracker.get_value()))**2 + 2.*9.8*(dot.get_center()[1]+3) )**.5 )/9.8, color=BLUE ))


        self.add(floor)
        self.play(ShowCreation(dot))
        self.play(ShowCreation(v0))
        self.play(dot.shift,2*UP+2*RIGHT)
        self.play(ShowCreation(traj))
        self.play(
            t_tracker.set_value,2*PI,
            rate_func=there_and_back,
            run_time=8
            )
    #TODO: make path dotted, increase v0, make particle move along path

class FClassicalBilliards(Scene): #*
    def construct(self):
        box = Box(6,6*16/9)

        ball = Dot(np.array([-2,-1,0]))

        self.add(box)
        self.wait(1)
        self.play(ShowCreation(ball))
        self.play(
            ball.shift, 3.5*np.array([-3,1,0])/np.linalg.norm(np.array([-3,1,0])),
            rate_func=linear,
            run_time=3.5/3
            )
        self.play(
            ball.shift, 9*np.array([3,1,0])/np.linalg.norm(np.array([3,1,0])),
            rate_func=linear,
            run_time=9/3
            )
        self.play(
            ball.shift, 2.2*np.array([3,-1,0])/np.linalg.norm(np.array([3,-1,0])),
            rate_func=linear,
            run_time=2.2/3
            )
        self.play(
            ball.shift, 9*np.array([-3,-1,0])/np.linalg.norm(np.array([-3,-1,0])),
            rate_func=linear,
            run_time=9/3
            )
       
class GParticleCloud(Scene):
    def construct(self):
        t = ValueTracker(1)

        box = Box(6,6*16/9)

        cloud = ProbabilityCloud(n_copies = 500, hbound = 3*16/9, vbound = 3)
        #cloud.move_to(2*LEFT)
        dot_gdw = cloud.gaussian_distribution_wrapper
        dot_gdw.add_updater(lambda v: v.set_width(t.get_value()))
        dot_gdw.rotate(TAU/8)

        self.add(box)
        self.add(cloud)
        self.play(
            ShowCreation(dot_gdw),
            dot_gdw.shift, 4*LEFT,
            rate_func=linear,
            run_time=.25
            )
        self.play(
            t.set_value,2,
            dot_gdw.shift, 6*RIGHT+UP,
            rate_func=linear,
            run_time=.5
            )
        self.play(
            t.set_value,2.25,
            dot_gdw.shift, 2*RIGHT+DOWN/2,
            rate_func=linear,
            run_time=.5/4
            )
        self.play(
            t.set_value,3,
            dot_gdw.move_to, ORIGIN,
            rate_func=linear,
            run_time=.5
            )
        self.play(
            t.set_value,5,
            rate_func=linear,
            run_time=1
            )

class HDistribution(Scene):
    def construct(self):
        tracker = ValueTracker(0)
        cloud = ProbabilityCloud(n_copies = 100, hbound = FRAME_WIDTH/2, vbound = FRAME_HEIGHT/2)
        dot_gdw = cloud.gaussian_distribution_wrapper
        #dot_gdw.set_width(2)
        #dot_gdw.rotate(TAU/4)
        gaussian = ParametricFunction(lambda x: np.array([x,np.exp(-x**2)+.1,0]),t_min = -FRAME_X_RADIUS,t_max = FRAME_X_RADIUS)
        self.add(cloud, dot_gdw)
        self.add(gaussian)
        self.play(tracker.set_value, 3, run_time = 5)

class IDeBroglie(Scene): #*
    def construct(self):
        debroglieform = TexMobject("p=\\hbar k")
        debroglieform.shift(2*LEFT+2*UP)
        uncertainty = TexMobject(r"\sigma_x\sigma_p\geq\frac{\hbar}{2}")
        uncertainty.shift(2*RIGHT+2*UP)
        k = ValueTracker(20)
        wave = self.get_wave(tr = k)
        dot1 = Dot(color= YELLOW_C)
        dot1.scale(6)
        dot1.shift(UP)
        dot2 = Dot(color= YELLOW_C)
        dot2.scale(1)
        dot2.shift(UP)

        self.add(wave,debroglieform,uncertainty,dot1)
        self.play(
            Transform(dot1,dot2),
            k.set_value, 5,
            rate_func = linear,
            run_time = 3
            )

        
    def get_wave(self,tr=None,**kwargs):
        def get_gaussian_func(x,c="r",t=0,k0=1,x0=0,t0=0, scale = 1):
            wave_part = np.exp(
                complex(0, ((k0+2*(t-t0)*(x-x0))*(x-x0)-k0**2*(t-t0)/2)/(1+4*(t-t0)**2))
            )
            bell_part = np.exp(-((x-x0)-k0*(t-t0))**2/(scale**2 * (1+4*(t-t0)**2)) )
            amplitude =  1 /( scale**(1/2) * (1 + 2j*(t-t0))**(1/2) )#scale * (2/PI)**1/4 / (1+2j*(t-t0))**1/2 
            if c == "r":
                return (amplitude*wave_part*bell_part).real-2
            if c == "i":
                return (amplitude*wave_part*bell_part).imag-2
        
        graphre = FunctionGraph(lambda x: get_gaussian_func(x))
        graphre.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="r",t=0, k0 = tr.get_value(), scale = 5/tr.get_value(),**kwargs)))
        
        graphim = FunctionGraph(lambda x: get_gaussian_func(x))
        graphim.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="i",t=0, k0 = tr.get_value(), scale = 5/tr.get_value(),**kwargs), color=BLUE))
        return VGroup(graphre,graphim)

class JAWaveFunc(Scene): #*
    def construct(self):
        t = ValueTracker(0)
        wavefunc = self.get_wave(tr = t,k0=10,x0 = 0)
        prob = self.get_abs(tr = t,k0=10,x0 = 0)

        wavetxt = TexMobject(
            r"\Psi(x,t)=",
            "u(x,t)",
            "+i",
            "v(x,t)"
            )
        wavetxt[1].set_color(color = YELLOW_C)
        wavetxt[3].set_color(color = BLUE)
        probtxt = TexMobject(
            r"|\Psi|^2",
            r"=\Psi^*\Psi=u^2+v^2")
        probtxt[0].set_color(color = RED)
        wavetxt.shift(3*UP)
        probtxt.shift(2*UP)

        self.play(ShowCreation(wavefunc),Write(wavetxt))

    def get_abs(self,tr=None,**kwargs):
        def get_gaussian_func(x,t=0,k0=1,x0=0,t0=0, scale = 1):
            bell_part = np.exp(-2*((x-x0)/scale-k0*(t-t0))**2/(1+4*(t-t0)**2))
            amplitude = 1 / (1+4*(t-t0))**(1/2) #scale * (2/PI)**1/2 / (1+4*(t-t0))**1/2
            return amplitude*bell_part
        
        graph = FunctionGraph(lambda x: get_gaussian_func(x))
        graph.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,t=tr.get_value(),**kwargs), color = RED))
        
        return graph

    def get_wave(self,tr=None,**kwargs):
        def get_gaussian_func(x,c="r",t=0,k0=1,x0=0,t0=0, scale = 1):
            wave_part = np.exp(
                complex(0, ((k0+2*(t-t0)*(x-x0)/scale)*(x-x0)/scale-k0**2*(t-t0)/2)/(1+4*(t-t0)**2))
            )
            bell_part = np.exp(-((x-x0)/scale-k0*(t-t0))**2/(1+4*(t-t0)**2))
            amplitude =  1 / (1 + 2j*(t-t0))**(1/2) #scale * (2/PI)**1/4 / (1+2j*(t-t0))**1/2 
            if c == "r":
                return (amplitude*wave_part*bell_part).real
            if c == "i":
                return (amplitude*wave_part*bell_part).imag
        
        graphre = FunctionGraph(lambda x: get_gaussian_func(x))
        graphre.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="r",t=tr.get_value(),**kwargs)))
        
        graphim = FunctionGraph(lambda x: get_gaussian_func(x))
        graphim.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="i",t=tr.get_value(),**kwargs), color=BLUE))
        return VGroup(graphre,graphim)

class JBWaveFunc(Scene): #*
    def construct(self):
        t = ValueTracker(0)
        wavefunc = self.get_wave(tr = t,k0=10,x0 = 0)
        prob = self.get_abs(tr = t,k0=10,x0 = 0)

        wavetxt = TexMobject(
            r"\Psi(x,t)=",
            "u(x,t)",
            "+i",
            "v(x,t)"
            )
        wavetxt[1].set_color(color = YELLOW_C)
        wavetxt[3].set_color(color = BLUE)
        probtxt = TexMobject(
            r"|\Psi|^2",
            r"=\Psi^*\Psi=u^2+v^2")
        probtxt[0].set_color(color = RED)
        wavetxt.shift(3*UP)
        probtxt.shift(2*UP)

        self.add(wavefunc,wavetxt)
        self.play(ShowCreation(prob),Write(probtxt))

    def get_abs(self,tr=None,**kwargs):
        def get_gaussian_func(x,t=0,k0=1,x0=0,t0=0, scale = 1):
            bell_part = np.exp(-2*((x-x0)/scale-k0*(t-t0))**2/(1+4*(t-t0)**2))
            amplitude = 1 / (1+4*(t-t0))**(1/2) #scale * (2/PI)**1/2 / (1+4*(t-t0))**1/2
            return amplitude*bell_part
        
        graph = FunctionGraph(lambda x: get_gaussian_func(x))
        graph.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,t=tr.get_value(),**kwargs), color = RED))
        
        return graph

    def get_wave(self,tr=None,**kwargs):
        def get_gaussian_func(x,c="r",t=0,k0=1,x0=0,t0=0, scale = 1):
            wave_part = np.exp(
                complex(0, ((k0+2*(t-t0)*(x-x0)/scale)*(x-x0)/scale-k0**2*(t-t0)/2)/(1+4*(t-t0)**2))
            )
            bell_part = np.exp(-((x-x0)/scale-k0*(t-t0))**2/(1+4*(t-t0)**2))
            amplitude =  1 / (1 + 2j*(t-t0))**(1/2) #scale * (2/PI)**1/4 / (1+2j*(t-t0))**1/2 
            if c == "r":
                return (amplitude*wave_part*bell_part).real
            if c == "i":
                return (amplitude*wave_part*bell_part).imag
        
        graphre = FunctionGraph(lambda x: get_gaussian_func(x,c="r",t=tr.get_value(),**kwargs))
        
        graphim = FunctionGraph(lambda x: get_gaussian_func(x,c="i",t=tr.get_value(),**kwargs), color=BLUE)
        return VGroup(graphre,graphim)

class JCWaveFunc(Scene): #*
    def construct(self):
        t = ValueTracker(0)
        wavefunc = self.get_wave(tr = t,k0=10,x0 = 0)
        prob = self.get_abs(tr = t,k0=10,x0 = 0)

        wavetxt = TexMobject(
            r"\Psi(x,t)=",
            "u(x,t)",
            "+i",
            "v(x,t)"
            )
        wavetxt[1].set_color(color = YELLOW_C)
        wavetxt[3].set_color(color = BLUE)
        probtxt = TexMobject(
            r"|\Psi|^2",
            r"=\Psi^*\Psi=u^2+v^2")
        probtxt[0].set_color(color = RED)
        wavetxt.shift(3*UP)
        probtxt.shift(2*UP)

        self.add(wavefunc,wavetxt,prob,probtxt)
        self.play(
            t.set_value,2,
            rate_func=linear,
            run_time=4
            )        

    def get_abs(self,tr=None,**kwargs):
        def get_gaussian_func(x,t=0,k0=1,x0=0,t0=0, scale = 1):
            bell_part = np.exp(-2*((x-x0)/scale-k0*(t-t0))**2/(1+4*(t-t0)**2))
            amplitude = 1 / (1+4*(t-t0))**(1/2) #scale * (2/PI)**1/2 / (1+4*(t-t0))**1/2
            return amplitude*bell_part
        
        graph = FunctionGraph(lambda x: get_gaussian_func(x))
        graph.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,t=tr.get_value(),**kwargs), color = RED))
        
        return graph

    def get_wave(self,tr=None,**kwargs):
        def get_gaussian_func(x,c="r",t=0,k0=1,x0=0,t0=0, scale = 1):
            wave_part = np.exp(
                complex(0, ((k0+2*(t-t0)*(x-x0)/scale)*(x-x0)/scale-k0**2*(t-t0)/2)/(1+4*(t-t0)**2))
            )
            bell_part = np.exp(-((x-x0)/scale-k0*(t-t0))**2/(1+4*(t-t0)**2))
            amplitude =  1 / (1 + 2j*(t-t0))**(1/2) #scale * (2/PI)**1/4 / (1+2j*(t-t0))**1/2 
            if c == "r":
                return (amplitude*wave_part*bell_part).real
            if c == "i":
                return (amplitude*wave_part*bell_part).imag
        
        graphre = FunctionGraph(lambda x: get_gaussian_func(x))
        graphre.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="r",t=tr.get_value(),**kwargs)))
        
        graphim = FunctionGraph(lambda x: get_gaussian_func(x))
        graphim.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="i",t=tr.get_value(),**kwargs), color=BLUE))
        return VGroup(graphre,graphim)

class KDiffyQs(Scene):
    def construct(self):
        Header = HeaderMobject("Differential Equations")
        bpt1 = TextMobject("-Language of mechanics")
        bpt2 = TextMobject("-Ordinary and partial")
        bpt3 = TextMobject("-Analytic vs numerical solutions")

        bpts = VGroup(bpt1,bpt2,bpt3)
        bpts.arrange_submobjects(DOWN)
        bpt2.align_to(bpt1,LEFT)
        bpt3.align_to(bpt1,LEFT)
        self.add(Header)
        self.play(Write(bpt1))
        self.wait(1)
        self.play(Write(bpt2))
        self.wait(1)
        self.play(Write(bpt3))
        self.wait(1)

class LAShowSlopes(GraphScene):
    CONFIG = {
        #"x_min": 0,
        "x_axis_label": "$t$",    
        "y_axis_label": "$f(t)$",
    }
    def construct(self):
        self.setup_axes()
        axes = self.axes
        func = self.get_graph(self.arbitrary_func)
        f = TexMobject(r"f(t)=")
        dfdt = TexMobject(r"\frac{\mathrm{d}f}{\mathrm{d}t}=")

        f.to_edge(UP)
        f.set_color(color=BLUE)
        dfdt.next_to(f,DOWN)
        dfdt.set_color(color=RED)
        
        t_tracker = ValueTracker(1.5)
        get_t = t_tracker.get_value

        ft_value = ValueTracker(self.arbitrary_func(t_tracker.get_value()))
        ftext = DecimalNumber(ft_value.get_value()).add_updater(lambda v: v.set_value(self.arbitrary_func(t_tracker.get_value())))
        ftext.next_to(f,RIGHT)
        ftext.set_color(color=BLUE)

        dfdtfunc = lambda x: -0.3*x.get_value()**2+1.8*x.get_value()-2.
        dfdtval = ValueTracker(dfdtfunc(t_tracker))
        dfdttext = DecimalNumber(dfdtval.get_value()).add_updater(lambda v: v.set_value(dfdtfunc(t_tracker)))
        dfdttext.next_to(dfdt,RIGHT)
        dfdttext.set_color(color=RED)
    
        def get_tangent_line(curve, alpha):
                line = Line(
                    ORIGIN, 5 * RIGHT,
                    color=RED,
                    stroke_width=3,
                )
                da = 0.0001
                p0 = curve.point_from_proportion(alpha)
                p1 = curve.point_from_proportion(alpha - da)
                p2 = curve.point_from_proportion(alpha + da)
                angle = angle_of_vector(p2 - p1)
                line.rotate(angle)
                line.move_to(p0)
                return line
    
        def get_slope_line():
                return get_tangent_line(
                    func, (get_t() + 1)/11
                )
        slope_line = always_redraw(get_slope_line)

        self.play(ShowCreation(func))
        self.play(Write(f),Write(ftext))
   
    def arbitrary_func(self,x):
        return -x**3/10.+0.9*x**2-2*x+5

class LBShowSlopes(GraphScene):
    CONFIG = {
        #"x_min": 0,
        "x_axis_label": "$t$",    
        "y_axis_label": "$f(t)$",
    }
    def construct(self):
        self.setup_axes()
        axes = self.axes
        func = self.get_graph(self.arbitrary_func)
        f = TexMobject(r"f(t)=")
        dfdt = TexMobject(r"\frac{\mathrm{d}f}{\mathrm{d}t}=")

        f.to_edge(UP)
        f.set_color(color=BLUE)
        dfdt.next_to(f,DOWN)
        dfdt.set_color(color=RED)
        
        t_tracker = ValueTracker(1.5)
        get_t = t_tracker.get_value

        ft_value = ValueTracker(self.arbitrary_func(t_tracker.get_value()))
        ftext = DecimalNumber(ft_value.get_value()).add_updater(lambda v: v.set_value(self.arbitrary_func(t_tracker.get_value())))
        ftext.next_to(f,RIGHT)
        ftext.set_color(color=BLUE)

        dfdtfunc = lambda x: -0.3*x.get_value()**2+1.8*x.get_value()-2.
        dfdtval = ValueTracker(dfdtfunc(t_tracker))
        dfdttext = DecimalNumber(dfdtval.get_value()).add_updater(lambda v: v.set_value(dfdtfunc(t_tracker)))
        dfdttext.next_to(dfdt,RIGHT)
        dfdttext.set_color(color=RED)
    
        def get_tangent_line(curve, alpha):
                line = Line(
                    ORIGIN, 5 * RIGHT,
                    color=RED,
                    stroke_width=3,
                )
                da = 0.0001
                p0 = curve.point_from_proportion(alpha)
                p1 = curve.point_from_proportion(alpha - da)
                p2 = curve.point_from_proportion(alpha + da)
                angle = angle_of_vector(p2 - p1)
                line.rotate(angle)
                line.move_to(p0)
                return line
    
        def get_slope_line():
                return get_tangent_line(
                    func, (get_t() + 1)/11
                )
        slope_line = always_redraw(get_slope_line)

        self.add(func, f, ftext)
        self.add(slope_line)
        self.play(
            ShowCreation(slope_line)
        )
        self.play(Write(dfdt),Write(dfdttext))
   
    def arbitrary_func(self,x):
        return -x**3/10.+0.9*x**2-2*x+5

class LCShowSlopes(GraphScene):
    CONFIG = {
        #"x_min": 0,
        "x_axis_label": "$t$",    
        "y_axis_label": "$f(t)$",
    }
    def construct(self):
        self.setup_axes()
        axes = self.axes
        func = self.get_graph(self.arbitrary_func)
        f = TexMobject(r"f(t)=")
        dfdt = TexMobject(r"\frac{\mathrm{d}f}{\mathrm{d}t}=")

        f.to_edge(UP)
        f.set_color(color=BLUE)
        dfdt.next_to(f,DOWN)
        dfdt.set_color(color=RED)
        
        t_tracker = ValueTracker(1.5)
        get_t = t_tracker.get_value

        ft_value = ValueTracker(self.arbitrary_func(t_tracker.get_value()))
        ftext = DecimalNumber(ft_value.get_value()).add_updater(lambda v: v.set_value(self.arbitrary_func(t_tracker.get_value())))
        ftext.next_to(f,RIGHT)
        ftext.set_color(color=BLUE)

        dfdtfunc = lambda x: -0.3*x.get_value()**2+1.8*x.get_value()-2.
        dfdtval = ValueTracker(dfdtfunc(t_tracker))
        dfdttext = DecimalNumber(dfdtval.get_value()).add_updater(lambda v: v.set_value(dfdtfunc(t_tracker)))
        dfdttext.next_to(dfdt,RIGHT)
        dfdttext.set_color(color=RED)
    
        def get_tangent_line(curve, alpha):
                line = Line(
                    ORIGIN, 5 * RIGHT,
                    color=RED,
                    stroke_width=3,
                )
                da = 0.0001
                p0 = curve.point_from_proportion(alpha)
                p1 = curve.point_from_proportion(alpha - da)
                p2 = curve.point_from_proportion(alpha + da)
                angle = angle_of_vector(p2 - p1)
                line.rotate(angle)
                line.move_to(p0)
                return line
    
        def get_slope_line():
                return get_tangent_line(
                    func, (get_t() + 1)/11
                )
        slope_line = always_redraw(get_slope_line)

        self.add(func, f, ftext, slope_line, dfdt, dfdttext)
        self.play(
            t_tracker.set_value,8,
            rate_func=wiggle,
            run_time=15
            )
   
    def arbitrary_func(self,x):
        return -x**3/10.+0.9*x**2-2*x+5

class MAShowSlopeExp(GraphScene):
    CONFIG = {
        #"x_min": 0,
        "x_max": 2,
        "x_axis_label": "$t$",    
        "y_axis_label": "$f(t)$",
    }
    def construct(self):
        self.setup_axes()
        axes = self.axes
        func = self.get_graph(self.arbitrary_func)
        f = TexMobject(r"f(t)=")
        dfdt = TexMobject(r"\frac{\mathrm{d}f}{\mathrm{d}t}=")
        ODE = TexMobject(r"\frac{\mathrm{d}f}{\mathrm{d}t}=f(t)")

        f.to_edge(UP)
        f.set_color(color=BLUE)
        dfdt.next_to(f,DOWN)
        dfdt.set_color(color=RED)
        ODE.to_edge(RIGHT)
        
        t_tracker = ValueTracker(0)
        get_t = t_tracker.get_value

        ft_value = ValueTracker(self.arbitrary_func(t_tracker.get_value()))
        ftext = DecimalNumber(ft_value.get_value()).add_updater(lambda v: v.set_value(self.arbitrary_func(t_tracker.get_value())))
        ftext.next_to(f,RIGHT)
        ftext.set_color(color=BLUE)

        dfdttext = ftext.copy()
        dfdttext.next_to(dfdt,RIGHT)
        dfdttext.set_color(color=RED)
    
        def get_tangent_line(curve, alpha):
                line = Line(
                    ORIGIN, 5 * RIGHT,
                    color=RED,
                    stroke_width=3,
                )
                da = 0.0001
                p0 = curve.point_from_proportion(alpha)
                p1 = curve.point_from_proportion(alpha - da)
                p2 = curve.point_from_proportion(alpha + da)
                angle = angle_of_vector(p2 - p1)
                line.rotate(angle)
                line.move_to(p0)
                return line
    
        def get_slope_line():
                return get_tangent_line(
                    func, (get_t() - self.x_min)/(self.x_max-self.x_min)
                )
        slope_line = always_redraw(get_slope_line)

        self.play(ShowCreation(func))
        self.play(Write(f),Write(ftext))
   
    def arbitrary_func(self,x):
        return np.exp(x)

class MBShowSlopeExp(GraphScene):
    CONFIG = {
        #"x_min": 0,
        "x_max": 2,
        "x_axis_label": "$t$",    
        "y_axis_label": "$f(t)$",
    }
    def construct(self):
        self.setup_axes()
        axes = self.axes
        func = self.get_graph(self.arbitrary_func)
        f = TexMobject(r"f(t)=")
        dfdt = TexMobject(r"\frac{\mathrm{d}f}{\mathrm{d}t}=")
        ODE = TexMobject(r"\frac{\mathrm{d}f}{\mathrm{d}t}=f(t)")

        f.to_edge(UP)
        f.set_color(color=BLUE)
        dfdt.next_to(f,DOWN)
        dfdt.set_color(color=RED)
        ODE.to_edge(RIGHT)
        
        t_tracker = ValueTracker(0)
        get_t = t_tracker.get_value

        ft_value = ValueTracker(self.arbitrary_func(t_tracker.get_value()))
        ftext = DecimalNumber(ft_value.get_value()).add_updater(lambda v: v.set_value(self.arbitrary_func(t_tracker.get_value())))
        ftext.next_to(f,RIGHT)
        ftext.set_color(color=BLUE)

        dfdttext = ftext.copy()
        dfdttext.next_to(dfdt,RIGHT)
        dfdttext.set_color(color=RED)
    
        def get_tangent_line(curve, alpha):
                line = Line(
                    ORIGIN, 5 * RIGHT,
                    color=RED,
                    stroke_width=3,
                )
                da = 0.0001
                p0 = curve.point_from_proportion(alpha)
                p1 = curve.point_from_proportion(alpha - da)
                p2 = curve.point_from_proportion(alpha + da)
                angle = angle_of_vector(p2 - p1)
                line.rotate(angle)
                line.move_to(p0)
                return line
    
        def get_slope_line():
                return get_tangent_line(
                    func, (get_t() - self.x_min)/(self.x_max-self.x_min)
                )
        slope_line = always_redraw(get_slope_line)

        self.add(func, f, ftext)
        self.add(slope_line)
        self.play(
            ShowCreation(slope_line)
        )
        self.play(Write(dfdt),Write(dfdttext))
   
    def arbitrary_func(self,x):
        return np.exp(x)

class MCShowSlopeExp(GraphScene):
    CONFIG = {
        #"x_min": 0,
        "x_max": 2,
        "x_axis_label": "$t$",    
        "y_axis_label": "$f(t)$",
    }
    def construct(self):
        self.setup_axes()
        axes = self.axes
        func = self.get_graph(self.arbitrary_func)
        f = TexMobject(r"f(t)=")
        dfdt = TexMobject(r"\frac{\mathrm{d}f}{\mathrm{d}t}=")
        ODE = TexMobject(r"\frac{\mathrm{d}f}{\mathrm{d}t}=f(t)")

        f.to_edge(UP)
        f.set_color(color=BLUE)
        dfdt.next_to(f,DOWN)
        dfdt.set_color(color=RED)
        ODE.to_edge(RIGHT)
        
        t_tracker = ValueTracker(0)
        get_t = t_tracker.get_value

        ft_value = ValueTracker(self.arbitrary_func(t_tracker.get_value()))
        ftext = DecimalNumber(ft_value.get_value()).add_updater(lambda v: v.set_value(self.arbitrary_func(t_tracker.get_value())))
        ftext.next_to(f,RIGHT)
        ftext.set_color(color=BLUE)

        dfdttext = ftext.copy()
        dfdttext.next_to(dfdt,RIGHT)
        dfdttext.set_color(color=RED)
    
        def get_tangent_line(curve, alpha):
                line = Line(
                    ORIGIN, 5 * RIGHT,
                    color=RED,
                    stroke_width=3,
                )
                da = 0.0001
                p0 = curve.point_from_proportion(alpha)
                p1 = curve.point_from_proportion(alpha - da)
                p2 = curve.point_from_proportion(alpha + da)
                angle = angle_of_vector(p2 - p1)
                line.rotate(angle)
                line.move_to(p0)
                return line
    
        def get_slope_line():
                return get_tangent_line(
                    func, (get_t() - self.x_min)/(self.x_max-self.x_min)
                )
        slope_line = always_redraw(get_slope_line)

        self.add(func, f, ftext, slope_line, dfdt, dfdttext)
        self.play(
            t_tracker.set_value,2,
            rate_func=wiggle,
            run_time=10
            )
   
    def arbitrary_func(self,x):
        return np.exp(x)

class MDShowSlopeExp(GraphScene):
    CONFIG = {
        #"x_min": 0,
        "x_max": 2,
        "x_axis_label": "$t$",    
        "y_axis_label": "$f(t)$",
    }
    def construct(self):
        self.setup_axes()
        axes = self.axes
        func = self.get_graph(self.arbitrary_func)
        f = TexMobject(r"f(t)=")
        dfdt = TexMobject(r"\frac{\mathrm{d}f}{\mathrm{d}t}=")
        ODE = TexMobject(r"\frac{\mathrm{d}f}{\mathrm{d}t}=f(t)")

        f.to_edge(UP)
        f.set_color(color=BLUE)
        dfdt.next_to(f,DOWN)
        dfdt.set_color(color=RED)
        ODE.to_edge(RIGHT)
        
        t_tracker = ValueTracker(0)
        get_t = t_tracker.get_value

        ft_value = ValueTracker(self.arbitrary_func(t_tracker.get_value()))
        ftext = DecimalNumber(ft_value.get_value()).add_updater(lambda v: v.set_value(self.arbitrary_func(t_tracker.get_value())))
        ftext.next_to(f,RIGHT)
        ftext.set_color(color=BLUE)

        dfdttext = ftext.copy()
        dfdttext.next_to(dfdt,RIGHT)
        dfdttext.set_color(color=RED)
    
        def get_tangent_line(curve, alpha):
                line = Line(
                    ORIGIN, 5 * RIGHT,
                    color=RED,
                    stroke_width=3,
                )
                da = 0.0001
                p0 = curve.point_from_proportion(alpha)
                p1 = curve.point_from_proportion(alpha - da)
                p2 = curve.point_from_proportion(alpha + da)
                angle = angle_of_vector(p2 - p1)
                line.rotate(angle)
                line.move_to(p0)
                return line
    
        def get_slope_line():
                return get_tangent_line(
                    func, (get_t() - self.x_min)/(self.x_max-self.x_min)
                )
        slope_line = always_redraw(get_slope_line)

        self.add(func,f,ftext,slope_line,dfdt,dfdttext)
        self.play(Transform(VGroup(f.copy(),dfdt.copy()),ODE))
   
    def arbitrary_func(self,x):
        return np.exp(x)

class NASpringODE(Scene):
    CONFIG = {
        "frequency" : 0.5,
        "ceiling_radius" : 3*FRAME_X_RADIUS,
        "n_springs" : 72,
        "amplitude" : 0.6,
        "spring_radius" : 0.15,
    }
    def construct(self):
        t = ValueTracker(PI/2)
        spring = self.create_spring(tr = t)
        ODE = TexMobject(r"\deriv{^2f}{t^2}=-\omega^2f")
        ODE.shift(2*UP)

        self.play(ShowCreation(spring))
        self.play(
            t.set_value, t.get_value()+2*PI,
            rate_func = linear,
            run_time=2
            )
        self.play(spring.shift, 4*LEFT,
            Write(ODE), 
            t.set_value, t.get_value()+2*PI,
            rate_func = linear,
            run_time=2
            )
        t0 = t.get_value()
        curve = ParametricFunction(lambda t: np.array([t,t,0]))
        curve.add_updater(lambda v: v.__init__(lambda x : np.array([x-3,-np.sin(t.get_value()-x),0]) ,
                t_min = 0, t_max = t.get_value()-t0+.01,
                color = BLUE,
                stroke_width = 2,) )
        self.add(curve)


    def create_spring(self, tr=None):
        floor = Line(LEFT,RIGHT)
        def get_spring(height = 1):
            t_max = 6.5
            r = self.spring_radius
            s = (height - r)/(t_max**2)
            spring = ParametricFunction(
                lambda t : op.add(
                    r*(np.sin(TAU*t)*RIGHT+np.cos(TAU*t)*UP),
                    s*((t_max - t)**2)*DOWN,
                ),
                t_min = 0, t_max = t_max,
                color = WHITE,
                stroke_width = 2,
            )
            spring.add_updater(lambda v: v.__init__(lambda t : op.add(
                    floor.get_center() + r*(np.sin(TAU*t)*RIGHT+np.cos(TAU*t)*UP),
                    (np.sin(tr.get_value()) - r)/(t_max**2)*((t_max - t)**2)*DOWN,
                ),
                t_min = 0, t_max = t_max,
                color = WHITE,
                stroke_width = 2,) )
            return spring
        height = tr
        spring = get_spring()
        square = Square(color=BLUE)
        square.set_fill(BLUE, opacity=0.5)
        square.scale(.3)
        square.add_updater(lambda v: v.move_to(floor.get_center()-np.sin(tr.get_value())*UP ))
        return VGroup(floor,spring,square)

class NBSpringODE(Scene):
    CONFIG = {
        "frequency" : 0.5,
        "ceiling_radius" : 3*FRAME_X_RADIUS,
        "n_springs" : 72,
        "amplitude" : 0.6,
        "spring_radius" : 0.15,
    }
    def construct(self):
        t = ValueTracker(PI/2)
        spring = self.create_spring(tr = t)
        spring.shift(4*LEFT)
        ODE = TexMobject(r"\deriv{^2f}{t^2}=-\omega^2f")
        ODE.shift(2*UP)

        self.add(spring, ODE)
        t0 = t.get_value()
        curve = ParametricFunction(lambda t: np.array([t,t,0]))
        curve.add_updater(lambda v: v.__init__(lambda x : np.array([x-3,-np.sin(t.get_value()-x),0]) ,
                t_min = 0, t_max = t.get_value()-t0+.01,
                color = BLUE,
                stroke_width = 2,) )
        self.add(curve)

        self.play(
            t.set_value, t.get_value()+8*PI,
            rate_func = linear,
            run_time=8
            )


    def create_spring(self, tr=None):
        floor = Line(LEFT,RIGHT)
        def get_spring(height = 1):
            t_max = 6.5
            r = self.spring_radius
            s = (height - r)/(t_max**2)
            spring = ParametricFunction(
                lambda t : op.add(
                    r*(np.sin(TAU*t)*RIGHT+np.cos(TAU*t)*UP),
                    s*((t_max - t)**2)*DOWN,
                ),
                t_min = 0, t_max = t_max,
                color = WHITE,
                stroke_width = 2,
            )
            spring.add_updater(lambda v: v.__init__(lambda t : op.add(
                    floor.get_center() + r*(np.sin(TAU*t)*RIGHT+np.cos(TAU*t)*UP),
                    (np.sin(tr.get_value()) - r)/(t_max**2)*((t_max - t)**2)*DOWN,
                ),
                t_min = 0, t_max = t_max,
                color = WHITE,
                stroke_width = 2,) )
            return spring
        height = tr
        spring = get_spring()
        square = Square(color=BLUE)
        square.set_fill(BLUE, opacity=0.5)
        square.scale(.3)
        square.add_updater(lambda v: v.move_to(floor.get_center()-np.sin(tr.get_value())*UP ))
        return VGroup(floor,spring,square)

class OAWavePDE(Scene): #*
    def construct(self):
        WaveEQN = TexMobject(r"\pderiv{^2u}{t^2}=c^2\pderiv{^2u}{x^2}")
        WaveEQN.shift(2*UP)

        t = ValueTracker(0)

        graph = FunctionGraph(lambda x: x)
        graph.add_updater(lambda v: v.__init__(lambda x: np.sin(x)*np.sin(2*t.get_value())-1 ))

        self.add(WaveEQN, graph)
        self.play(
            t.set_value, 2*PI,
            rate_func=linear,
            run_time=2*PI
            )

class OBWavePDE(Scene): #*
    def construct(self):
        WaveEQN = TexMobject(r"\pderiv{^2u}{t^2}=c^2\pderiv{^2u}{x^2}")
        WaveEQN.shift(2*UP)

        t = ValueTracker(0)

        graph = FunctionGraph(lambda x: x)
        graph.add_updater(lambda v: v.__init__(lambda x: np.sin(x-2*t.get_value())-1 ))

        self.add(WaveEQN, graph)
        self.play(
            t.set_value, 2*PI,
            rate_func=linear,
            run_time=2*PI
            )        

class PAShowSE(Scene):
    def construct(self):
        SE = TexMobject(
            "i\\hbar\\pderiv{\\Psi}{t}", #0
            "=", #1
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi}{x^2}", #2
            "+", #3
            "V\\Psi" #4
            )
        SEEnergy = TexMobject(
            "\\underbrace{\\qquad}_E", #0
            "\\underbrace{\\qquad\\quad}_K", #1
            "\\underbrace{\\qquad}_V" #2
            )
        FreeParticle = TexMobject(
            "\\text{free particle }V=0"
            )
        Seperable = TexMobject(
            "\\text{stationary states }\\Psi(x,t)=\\psi(x)\\phi(t)"
            )
        TISE = TexMobject(
            "-\\frac{\\hbar^2}{2m}\\deriv{^2\\psi}{x^2}=E\\psi"
            )
        TDSE = TexMobject(
            "i\\hbar\\deriv{\\phi}{t}=E\\phi"
            )
        phi = TexMobject(
            "\\phi(t)=e^{i\\omega t}"
            )
        psi = TexMobject(
            "\\psi(x)=e^{ikx}"
            )


        FreeParticle.shift(2*DOWN)
        Seperable.next_to(FreeParticle,DOWN)
        SEEnergy[0].next_to(SE[0],DOWN)
        SEEnergy[1].next_to(SE[2],DOWN)
        SEEnergy[2].next_to(SE[4],1.5*DOWN)
        TISE.shift(2*RIGHT)
        TDSE.shift(2*LEFT)
        phi.shift(2*LEFT)
        psi.shift(2*RIGHT)

        self.play(Write(SE[0:5]))

class PBShowSE(Scene):
    def construct(self):
        SE = TexMobject(
            "i\\hbar\\pderiv{\\Psi}{t}", #0
            "=", #1
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi}{x^2}", #2
            "+", #3
            "V\\Psi" #4
            )
        SEEnergy = TexMobject(
            "\\underbrace{\\qquad}_E", #0
            "\\underbrace{\\qquad\\quad}_K", #1
            "\\underbrace{\\qquad}_V" #2
            )
        FreeParticle = TexMobject(
            "\\text{free particle }V=0"
            )
        Seperable = TexMobject(
            "\\text{stationary states }\\Psi(x,t)=\\psi(x)\\phi(t)"
            )
        TISE = TexMobject(
            "-\\frac{\\hbar^2}{2m}\\deriv{^2\\psi}{x^2}=E\\psi"
            )
        TDSE = TexMobject(
            "i\\hbar\\deriv{\\phi}{t}=E\\phi"
            )
        phi = TexMobject(
            "\\phi(t)=e^{i\\omega t}"
            )
        psi = TexMobject(
            "\\psi(x)=e^{ikx}"
            )


        FreeParticle.shift(2*DOWN)
        Seperable.next_to(FreeParticle,DOWN)
        SEEnergy[0].next_to(SE[0],DOWN)
        SEEnergy[1].next_to(SE[2],DOWN)
        SEEnergy[2].next_to(SE[4],1.5*DOWN)
        TISE.shift(2*RIGHT)
        TDSE.shift(2*LEFT)
        phi.shift(2*LEFT)
        psi.shift(2*RIGHT)

        self.add(SE[0:5])
        #self.wait(2)
        self.play(Write(SEEnergy[0]))
        #self.wait(2)
        self.play(ShowCreation(SEEnergy[1]))
        #self.wait(2)
        self.play(Write(SEEnergy[2]))

class PCShowSE(Scene):
    def construct(self):
        SE = TexMobject(
            "i\\hbar\\pderiv{\\Psi}{t}", #0
            "=", #1
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi}{x^2}", #2
            "+", #3
            "V\\Psi" #4
            )
        SEEnergy = TexMobject(
            "\\underbrace{\\qquad}_E", #0
            "\\underbrace{\\qquad\\quad}_K", #1
            "\\underbrace{\\qquad}_V" #2
            )
        FreeParticle = TexMobject(
            "\\text{free particle }V=0"
            )
        Seperable = TexMobject(
            "\\text{stationary states }\\Psi(x,t)=\\psi(x)\\phi(t)"
            )
        TISE = TexMobject(
            "-\\frac{\\hbar^2}{2m}\\deriv{^2\\psi}{x^2}=E\\psi"
            )
        TDSE = TexMobject(
            "i\\hbar\\deriv{\\phi}{t}=E\\phi"
            )
        phi = TexMobject(
            "\\phi(t)=e^{i\\omega t}"
            )
        psi = TexMobject(
            "\\psi(x)=e^{ikx}"
            )


        FreeParticle.shift(2*DOWN)
        Seperable.next_to(FreeParticle,DOWN)
        SEEnergy[0].next_to(SE[0],DOWN)
        SEEnergy[1].next_to(SE[2],DOWN)
        SEEnergy[2].next_to(SE[4],1.5*DOWN)
        TISE.shift(2*RIGHT)
        TDSE.shift(2*LEFT)
        phi.shift(2*LEFT)
        psi.shift(2*RIGHT)

        self.add(SE[0:5],SEEnergy)
        self.play(Write(FreeParticle))
        self.play(FadeOut(SE[3:5]),FadeOut(SEEnergy[2]),SE[0:3].shift, RIGHT,SEEnergy[0:2].shift, RIGHT)

class PDShowSE(Scene):
    def construct(self):
        SE = TexMobject(
            "i\\hbar\\pderiv{\\Psi}{t}", #0
            "=", #1
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi}{x^2}", #2
            "+", #3
            "V\\Psi" #4
            )
        SEEnergy = TexMobject(
            "\\underbrace{\\qquad}_E", #0
            "\\underbrace{\\qquad\\quad}_K", #1
            "\\underbrace{\\qquad}_V" #2
            )
        FreeParticle = TexMobject(
            "\\text{free particle }V=0"
            )
        Seperable = TexMobject(
            "\\text{stationary states }\\Psi(x,t)=\\psi(x)\\phi(t)"
            )
        TISE = TexMobject(
            "-\\frac{\\hbar^2}{2m}\\deriv{^2\\psi}{x^2}=E\\psi"
            )
        TDSE = TexMobject(
            "i\\hbar\\deriv{\\phi}{t}=E\\phi"
            )
        phi = TexMobject(
            "\\phi(t)=e^{i\\omega t}"
            )
        psi = TexMobject(
            "\\psi(x)=e^{ikx}"
            )


        FreeParticle.shift(2*DOWN)
        Seperable.next_to(FreeParticle,DOWN)
        SEEnergy[0].next_to(SE[0],DOWN)
        SEEnergy[1].next_to(SE[2],DOWN)
        SEEnergy[2].next_to(SE[4],1.5*DOWN)
        TISE.shift(2*RIGHT)
        TDSE.shift(2*LEFT)
        phi.shift(2*LEFT)
        psi.shift(2*RIGHT)
        SE[0:3].shift(RIGHT)
        SEEnergy[0:2].shift(RIGHT)

        self.add(SE[0:3],SEEnergy[0:2],FreeParticle)
        self.play(Write(Seperable),FadeOut(SEEnergy[0:2]))

class PEShowSE(Scene):
    def construct(self):
        SE = TexMobject(
            "i\\hbar\\pderiv{\\Psi}{t}", #0
            "=", #1
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi}{x^2}", #2
            "+", #3
            "V\\Psi" #4
            )
        SEEnergy = TexMobject(
            "\\underbrace{\\qquad}_E", #0
            "\\underbrace{\\qquad\\quad}_K", #1
            "\\underbrace{\\qquad}_V" #2
            )
        FreeParticle = TexMobject(
            "\\text{free particle }V=0"
            )
        Seperable = TexMobject(
            "\\text{stationary states }\\Psi(x,t)=\\psi(x)\\phi(t)"
            )
        TISE = TexMobject(
            "-\\frac{\\hbar^2}{2m}\\deriv{^2\\psi}{x^2}=E\\psi"
            )
        TDSE = TexMobject(
            "i\\hbar\\deriv{\\phi}{t}=E\\phi"
            )
        phi = TexMobject(
            "\\phi(t)=e^{i\\omega t}"
            )
        psi = TexMobject(
            "\\psi(x)=e^{ikx}"
            )


        FreeParticle.shift(2*DOWN)
        Seperable.next_to(FreeParticle,DOWN)
        SEEnergy[0].next_to(SE[0],DOWN)
        SEEnergy[1].next_to(SE[2],DOWN)
        SEEnergy[2].next_to(SE[4],1.5*DOWN)
        TISE.shift(2*RIGHT)
        TDSE.shift(2*LEFT)
        phi.shift(2*LEFT)
        psi.shift(2*RIGHT)
        SE[0:3].shift(RIGHT)
        SEEnergy[0:2].shift(RIGHT)

        self.add(SE[0:3],FreeParticle,Seperable)
        self.play(ReplacementTransform(SE[0:2],TDSE),ReplacementTransform(SE[2:3],TISE))

class PFShowSE(Scene):
    def construct(self):
        SE = TexMobject(
            "i\\hbar\\pderiv{\\Psi}{t}", #0
            "=", #1
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi}{x^2}", #2
            "+", #3
            "V\\Psi" #4
            )
        SEEnergy = TexMobject(
            "\\underbrace{\\qquad}_E", #0
            "\\underbrace{\\qquad\\quad}_K", #1
            "\\underbrace{\\qquad}_V" #2
            )
        FreeParticle = TexMobject(
            "\\text{free particle }V=0"
            )
        Seperable = TexMobject(
            "\\text{stationary states }\\Psi(x,t)=\\psi(x)\\phi(t)"
            )
        TISE = TexMobject(
            "-\\frac{\\hbar^2}{2m}\\deriv{^2\\psi}{x^2}=E\\psi"
            )
        TDSE = TexMobject(
            "i\\hbar\\deriv{\\phi}{t}=E\\phi"
            )
        phi = TexMobject(
            "\\phi(t)=e^{i\\omega t}"
            )
        psi = TexMobject(
            "\\psi(x)=e^{ikx}"
            )


        FreeParticle.shift(2*DOWN)
        Seperable.next_to(FreeParticle,DOWN)
        SEEnergy[0].next_to(SE[0],DOWN)
        SEEnergy[1].next_to(SE[2],DOWN)
        SEEnergy[2].next_to(SE[4],1.5*DOWN)
        TISE.shift(2*RIGHT)
        TDSE.shift(2*LEFT)
        phi.shift(2*LEFT)
        psi.shift(2*RIGHT)
        SE[0:3].shift(RIGHT)
        SEEnergy[0:2].shift(RIGHT)

        self.add(TDSE,TISE,FreeParticle,Seperable)

        self.play(TDSE.shift, 2*UP, FadeIn(phi))

class PGShowSE(Scene):
    def construct(self):
        SE = TexMobject(
            "i\\hbar\\pderiv{\\Psi}{t}", #0
            "=", #1
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi}{x^2}", #2
            "+", #3
            "V\\Psi" #4
            )
        SEEnergy = TexMobject(
            "\\underbrace{\\qquad}_E", #0
            "\\underbrace{\\qquad\\quad}_K", #1
            "\\underbrace{\\qquad}_V" #2
            )
        FreeParticle = TexMobject(
            "\\text{free particle }V=0"
            )
        Seperable = TexMobject(
            "\\text{stationary states }\\Psi(x,t)=\\psi(x)\\phi(t)"
            )
        TISE = TexMobject(
            "-\\frac{\\hbar^2}{2m}\\deriv{^2\\psi}{x^2}=E\\psi"
            )
        TDSE = TexMobject(
            "i\\hbar\\deriv{\\phi}{t}=E\\phi"
            )
        phi = TexMobject(
            "\\phi(t)=e^{i\\omega t}"
            )
        psi = TexMobject(
            "\\psi(x)=e^{ikx}"
            )


        FreeParticle.shift(2*DOWN)
        Seperable.next_to(FreeParticle,DOWN)
        SEEnergy[0].next_to(SE[0],DOWN)
        SEEnergy[1].next_to(SE[2],DOWN)
        SEEnergy[2].next_to(SE[4],1.5*DOWN)
        TISE.shift(2*RIGHT)
        TDSE.shift(2*LEFT)
        phi.shift(2*LEFT)
        psi.shift(2*RIGHT)
        SE[0:3].shift(RIGHT)
        SEEnergy[0:2].shift(RIGHT)
        TDSE.shift(2*UP)

        self.add(TDSE,TISE,FreeParticle,Seperable,phi)
        self.play(TISE.shift, 2*UP, FadeIn(psi))

class QStationaryStates(Scene):
    def construct(self):
        t_tracker = ValueTracker(0)

        k = ValueTracker(1)
        ktext = TexMobject("k=")
        kval = DecimalNumber(k.get_value()).add_updater(lambda v: v.set_value(k.get_value()))
        kval.next_to(ktext)
        kval.shift(3*DOWN)
        ktext.shift(3*DOWN)

        Psitext = TexMobject(
            "\\Psi(x,t)=e^{ik(x-\\frac{\\hbar k}{2m}t)}=\\cos(k(x-\\frac{\\hbar k}{2m}t))+i\\sin(k(x-\\frac{\\hbar k}{2m}t))"
            )
        Psitext.shift(3*UP)
        
        psi = self.get_wave(ttr=t_tracker,ktr=k,x0=0, scale = 1)

        self.play(ShowCreation(psi),Write(Psitext))
        self.play(Write(ktext),Write(kval))
        self.play(
            t_tracker.set_value,1,
            rate_func=linear,
            run_time=1
            )
        self.wait(1)
        self.play(
            k.set_value,5,
            t_tracker.set_value,0,
            rate_func=linear,
            run_time=1,
            )
        self.wait(1)
        self.play(
            t_tracker.set_value,1,
            rate_func=linear,
            run_time=1
            )
        self.wait(1)
        self.play(
            k.set_value,-1,
            t_tracker.set_value,0,
            rate_func=linear,
            run_time=1,
            )
        self.wait(1)
        self.play(
            t_tracker.set_value,1,
            rate_func=linear,
            run_time=1
            )
        

    def get_wave(self,ttr=None,ktr=None,**kwargs):
        def get_gaussian_func(x,c="r",t=0,k=1,x0=0,t0=0, scale = 1):
            wave_part = np.exp(
                complex(0, k*(x-x0-k*(t-t0)))
            )
            amplitude = scale #(2/PI)**1/4 / (1+2j*(t-t0))**1/2
            if c == "r":
                return (amplitude*wave_part).real
            if c == "i":
                return (amplitude*wave_part).imag
        
        graphre = FunctionGraph(lambda x: get_gaussian_func(x))
        graphre.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="r",t=ttr.get_value(),k=ktr.get_value(),**kwargs)))
        
        graphim = FunctionGraph(lambda x: get_gaussian_func(x))
        graphim.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="i",t=ttr.get_value(),k=ktr.get_value(),**kwargs), color=BLUE))
        return VGroup(graphre,graphim)

class RALinearity(Scene):
    def construct(self):
        Linearity = HeaderMobject("Linearity")
        Lop = TexMobject(
            "L", #0
            "(", #1
            "c_1", #2
            "\\Psi_1", #3
            "+", #4
            "c_2", #5
            "\\Psi_2", #6
            ")", #7
            "=", #8
            "c_1", #9
            "L", #10
            "(", #11
            "\\Psi_1", #12
            ")", #13
            "+", #14
            "c_2", #15
            "L", #16
            "(", #17
            "\\Psi_2", #18
            ")", #19
            )
        LD = TexMobject(
            "\\pderiv{}{x}", #0
            "(", #1
            "c_1", #2
            "\\Psi_1", #3
            "+", #4
            "c_2", #5
            "\\Psi_2", #6
            ")", #7
            "=", #8
            "c_1", #9
            "\\pderiv{}{x}", #10
            "(", #11
            "\\Psi_1", #12
            ")", #13
            "+", #14
            "c_2", #15
            "\\pderiv{}{x}", #16
            "(", #17
            "\\Psi_2", #18
            ")", #19
            )
        SE1 = TexMobject(
            "i\\hbar\\pderiv{\\Psi_1}{t}", #0
            "=", #1
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi_1}{x^2}", #2
            )
        SE2 = TexMobject(
            "i\\hbar\\pderiv{\\Psi_2}{t}", #0
            "=", #1
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi_2}{x^2}", #2
            )
        SE12 = TexMobject(
            "i\\hbar\\pderiv{\\Psi_1}{t}", #0
            "+", #1
            "i\\hbar\\pderiv{\\Psi_2}{t}", #2
            "=", #3
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi_2}{x^2}", #4
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi_2}{x^2}", #5
            )
        DSE12 = TexMobject(
            "i\\hbar\\pderiv{}{t}", #0
            "(\\Psi_1+\\Psi_2)", #1
            "=", #2
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2}{x^2}", #3
            "(\\Psi_1+\\Psi_2)", #4
            )

        SE1.shift(UP+3*LEFT)
        SE2.shift(UP+3*RIGHT)
        SE12.shift(DOWN)
        DSE12.shift(DOWN)
        self.add(Linearity)
        self.play(Write(Lop[0:9]))
        self.play(
            ReplacementTransform(Lop[0].copy(),Lop[10]),
            ReplacementTransform(Lop[0].copy(),Lop[16]),         
            ReplacementTransform(Lop[1].copy(),Lop[11]),
            ReplacementTransform(Lop[1].copy(),Lop[17]),
            ReplacementTransform(Lop[2].copy(),Lop[9]),
            ReplacementTransform(Lop[3].copy(),Lop[12]),
            ReplacementTransform(Lop[4].copy(),Lop[14]),
            ReplacementTransform(Lop[5].copy(),Lop[15]),
            ReplacementTransform(Lop[6].copy(),Lop[18]),
            ReplacementTransform(Lop[7].copy(),Lop[13]),
            ReplacementTransform(Lop[7].copy(),Lop[19]),
            )

class RBLinearity(Scene):
    def construct(self):
        Linearity = HeaderMobject("Linearity")
        Lop = TexMobject(
            "L", #0
            "(", #1
            "c_1", #2
            "\\Psi_1", #3
            "+", #4
            "c_2", #5
            "\\Psi_2", #6
            ")", #7
            "=", #8
            "c_1", #9
            "L", #10
            "(", #11
            "\\Psi_1", #12
            ")", #13
            "+", #14
            "c_2", #15
            "L", #16
            "(", #17
            "\\Psi_2", #18
            ")", #19
            )
        LD = TexMobject(
            "\\pderiv{}{x}", #0
            "(", #1
            "c_1", #2
            "\\Psi_1", #3
            "+", #4
            "c_2", #5
            "\\Psi_2", #6
            ")", #7
            "=", #8
            "c_1", #9
            "\\pderiv{}{x}", #10
            "(", #11
            "\\Psi_1", #12
            ")", #13
            "+", #14
            "c_2", #15
            "\\pderiv{}{x}", #16
            "(", #17
            "\\Psi_2", #18
            ")", #19
            )
        SE1 = TexMobject(
            "i\\hbar\\pderiv{\\Psi_1}{t}", #0
            "=", #1
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi_1}{x^2}", #2
            )
        SE2 = TexMobject(
            "i\\hbar\\pderiv{\\Psi_2}{t}", #0
            "=", #1
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi_2}{x^2}", #2
            )
        SE12 = TexMobject(
            "i\\hbar\\pderiv{\\Psi_1}{t}", #0
            "+", #1
            "i\\hbar\\pderiv{\\Psi_2}{t}", #2
            "=", #3
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi_2}{x^2}", #4
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi_2}{x^2}", #5
            )
        DSE12 = TexMobject(
            "i\\hbar\\pderiv{}{t}", #0
            "(\\Psi_1+\\Psi_2)", #1
            "=", #2
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2}{x^2}", #3
            "(\\Psi_1+\\Psi_2)", #4
            )

        SE1.shift(UP+3*LEFT)
        SE2.shift(UP+3*RIGHT)
        SE12.shift(DOWN)
        DSE12.shift(DOWN)
        self.add(Linearity, Lop)
        self.play(FadeOut(Lop))
        self.play(Write(LD[0:9]))
        self.play(
            ReplacementTransform(LD[0].copy(),LD[10]),
            ReplacementTransform(LD[0].copy(),LD[16]),         
            ReplacementTransform(LD[1].copy(),LD[11]),
            ReplacementTransform(LD[1].copy(),LD[17]),
            ReplacementTransform(LD[2].copy(),LD[9]),
            ReplacementTransform(LD[3].copy(),LD[12]),
            ReplacementTransform(LD[4].copy(),LD[14]),
            ReplacementTransform(LD[5].copy(),LD[15]),
            ReplacementTransform(LD[6].copy(),LD[18]),
            ReplacementTransform(LD[7].copy(),LD[13]),
            ReplacementTransform(LD[7].copy(),LD[19]),
            )

class RCLinearity(Scene):
    def construct(self):
        Linearity = HeaderMobject("Linearity")
        Lop = TexMobject(
            "L", #0
            "(", #1
            "c_1", #2
            "\\Psi_1", #3
            "+", #4
            "c_2", #5
            "\\Psi_2", #6
            ")", #7
            "=", #8
            "c_1", #9
            "L", #10
            "(", #11
            "\\Psi_1", #12
            ")", #13
            "+", #14
            "c_2", #15
            "L", #16
            "(", #17
            "\\Psi_2", #18
            ")", #19
            )
        LD = TexMobject(
            "\\pderiv{}{x}", #0
            "(", #1
            "c_1", #2
            "\\Psi_1", #3
            "+", #4
            "c_2", #5
            "\\Psi_2", #6
            ")", #7
            "=", #8
            "c_1", #9
            "\\pderiv{}{x}", #10
            "(", #11
            "\\Psi_1", #12
            ")", #13
            "+", #14
            "c_2", #15
            "\\pderiv{}{x}", #16
            "(", #17
            "\\Psi_2", #18
            ")", #19
            )
        SE1 = TexMobject(
            "i\\hbar\\pderiv{\\Psi_1}{t}", #0
            "=", #1
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi_1}{x^2}", #2
            )
        SE2 = TexMobject(
            "i\\hbar\\pderiv{\\Psi_2}{t}", #0
            "=", #1
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi_2}{x^2}", #2
            )
        SE12 = TexMobject(
            "i\\hbar\\pderiv{\\Psi_1}{t}", #0
            "+", #1
            "i\\hbar\\pderiv{\\Psi_2}{t}", #2
            "=", #3
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi_2}{x^2}", #4
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi_2}{x^2}", #5
            )
        DSE12 = TexMobject(
            "i\\hbar\\pderiv{}{t}", #0
            "(\\Psi_1+\\Psi_2)", #1
            "=", #2
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2}{x^2}", #3
            "(\\Psi_1+\\Psi_2)", #4
            )

        SE1.shift(UP+3*LEFT)
        SE2.shift(UP+3*RIGHT)
        SE12.shift(DOWN)
        DSE12.shift(DOWN)
        self.add(Linearity,LD)
        self.play(FadeOut(LD))
        self.play(ShowCreation(SE1))
        self.play(ShowCreation(SE2))
        self.play(
            ReplacementTransform(VGroup(SE1[0].copy(),SE2[0].copy()),SE12[0:3]),
            ReplacementTransform(VGroup(SE1[1].copy(),SE2[1].copy()),SE12[3]),
            ReplacementTransform(VGroup(SE1[2].copy(),SE2[2].copy()),SE12[4:6])
            )
        self.remove(SE12)

class RDLinearity(Scene):
    def construct(self):
        Linearity = HeaderMobject("Linearity")
        Lop = TexMobject(
            "L", #0
            "(", #1
            "c_1", #2
            "\\Psi_1", #3
            "+", #4
            "c_2", #5
            "\\Psi_2", #6
            ")", #7
            "=", #8
            "c_1", #9
            "L", #10
            "(", #11
            "\\Psi_1", #12
            ")", #13
            "+", #14
            "c_2", #15
            "L", #16
            "(", #17
            "\\Psi_2", #18
            ")", #19
            )
        LD = TexMobject(
            "\\pderiv{}{x}", #0
            "(", #1
            "c_1", #2
            "\\Psi_1", #3
            "+", #4
            "c_2", #5
            "\\Psi_2", #6
            ")", #7
            "=", #8
            "c_1", #9
            "\\pderiv{}{x}", #10
            "(", #11
            "\\Psi_1", #12
            ")", #13
            "+", #14
            "c_2", #15
            "\\pderiv{}{x}", #16
            "(", #17
            "\\Psi_2", #18
            ")", #19
            )
        SE1 = TexMobject(
            "i\\hbar\\pderiv{\\Psi_1}{t}", #0
            "=", #1
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi_1}{x^2}", #2
            )
        SE2 = TexMobject(
            "i\\hbar\\pderiv{\\Psi_2}{t}", #0
            "=", #1
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi_2}{x^2}", #2
            )
        SE12 = TexMobject(
            "i\\hbar\\pderiv{\\Psi_1}{t}", #0
            "+", #1
            "i\\hbar\\pderiv{\\Psi_2}{t}", #2
            "=", #3
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi_2}{x^2}", #4
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi_2}{x^2}", #5
            )
        DSE12 = TexMobject(
            "i\\hbar\\pderiv{}{t}", #0
            "(\\Psi_1+\\Psi_2)", #1
            "=", #2
            "-\\frac{\\hbar^2}{2m}\\pderiv{^2}{x^2}", #3
            "(\\Psi_1+\\Psi_2)", #4
            )

        SE1.shift(UP+3*LEFT)
        SE2.shift(UP+3*RIGHT)
        SE12.shift(DOWN)
        DSE12.shift(DOWN)
        self.add(Linearity,SE1,SE2,SE12)
        self.play(
            ReplacementTransform(SE12[0],DSE12[0]),
            ReplacementTransform(SE12[1:3],DSE12[1]),
            ReplacementTransform(SE12[3],DSE12[2]),
            ReplacementTransform(SE12[4],DSE12[3]),
            ReplacementTransform(SE12[5],DSE12[4]),
            )

class SNonLinearity(Scene): #*
    def construct(self):
        Linearity = HeaderMobject("What is not Linear?")
        Abs = TexMobject(
            "|", #0
            "c_1", #1
            "\\Psi_1", #2
            "+", #3
            "c_2", #4
            "\\Psi_2", #5
            "|^2", #6
            "=", #7
            "|", #8
            "c_1", #9
            "|^2", #10
            "|", #11
            "\\Psi_1", #12
            "|^2", #13
            "+", #14
            "c_1^*", #15
            "c_2", #16
            "\\Psi_1^*", #17
            "\\Psi_2", #18
            "+", #19
            "c_1", #20
            "c_2^*", #21
            "\\Psi_1", #22
            "\\Psi_2^*", #23
            "+", #24
            "|", #25
            "c_2", #26
            "|^2", #27
            "|", #28
            "\\Psi_2", #29
            "|^2" #30
            )
        NotAbs = TexMobject(r"\neq c_1|\Psi_1|^2+c_2|\Psi_2|^2")
        NotAbs.next_to(Abs[7],1.5*DOWN)
        NotAbs.shift((1.95)*RIGHT)

        self.add(Linearity)
        self.play(Write(Abs[0:8]))
        self.play(
            ReplacementTransform(Abs[0].copy(),Abs[8]),
            ReplacementTransform(Abs[0].copy(),Abs[11]),
            ReplacementTransform(Abs[0].copy(),Abs[25]),
            ReplacementTransform(Abs[0].copy(),Abs[28]),
            ReplacementTransform(Abs[1].copy(),Abs[9]),
            ReplacementTransform(Abs[2].copy(),Abs[12]),
            ReplacementTransform(Abs[3].copy(),Abs[24]),
            ReplacementTransform(Abs[4].copy(),Abs[26]),
            ReplacementTransform(Abs[5].copy(),Abs[29]),
            ReplacementTransform(Abs[6].copy(),Abs[10]),
            ReplacementTransform(Abs[6].copy(),Abs[13]),
            ReplacementTransform(Abs[6].copy(),Abs[27]),
            ReplacementTransform(Abs[6].copy(),Abs[30])
            )
        self.play(
            ReplacementTransform(Abs[1].copy(),Abs[15]),
            ReplacementTransform(Abs[1].copy(),Abs[20]),
            ReplacementTransform(Abs[2].copy(),Abs[17]),
            ReplacementTransform(Abs[2].copy(),Abs[22]),
            ReplacementTransform(Abs[3].copy(),Abs[14]),
            ReplacementTransform(Abs[3].copy(),Abs[19]),
            ReplacementTransform(Abs[4].copy(),Abs[16]),
            ReplacementTransform(Abs[4].copy(),Abs[21]),
            ReplacementTransform(Abs[4].copy(),Abs[16]),
            ReplacementTransform(Abs[5].copy(),Abs[18]),
            ReplacementTransform(Abs[5].copy(),Abs[23]),
            )
        self.play(Write(NotAbs))

class TFourierWaves(Scene): #*
    def construct(self):
        t = ValueTracker(0)

        spacing = 1
        center = -1.2
        lpsi2 = self.get_wave(ttr = t, k = -2, scale = .4, shift = (-spacing*1.5+center)*UP)
        lpsi1 = self.get_wave(ttr = t, k = -1, scale = .4, shift = (-spacing*.5+center)*UP)
        rpsi1 = self.get_wave(ttr = t, k = 1, scale = .4, shift = (spacing*.5+center)*UP)
        rpsi2 = self.get_wave(ttr = t, k = 2, scale = .4, shift = (spacing*1.5+center)*UP)
        psisum = self.get_wave_sum(ttr = t, shift = 2*UP, scale = .1)

        #lpsi2.shift((-spacing*2+center)*UP)
        #lpsi1.shift((-spacing*1+center)*UP)
        #rpsi1.shift((spacing*1+center)*UP)
        #rpsi2.shift((spacing*2+center)*UP)

        self.add(
            lpsi2,
            lpsi1,
            rpsi1,
            rpsi2,
            psisum)
        self.play(
            t.set_value,2*PI,
            rate_func=linear,
            run_time=2*PI
            )
    #    self.wait(1)
    #    self.play(
    #        #,rpsi1.copy(),rpsi2.copy()
    #        Transform(VGroup(lpsi2.copy(),lpsi1.copy(),rpsi1.copy(),rpsi2.copy()),psisum))
    #    self.wait(1)
    #    self.play(
    #        t.set_value,2*PI,
    #        rate_func=linear,
    #        run_time=2*PI
    #        )


    def get_wave(self,ttr=None, shift = 0,**kwargs):
        def get_gaussian_func(x,c="r",t=0,k=1,x0=0,t0=0, scale = 1):
            wave_part = np.exp(
                complex(0, k*(x-x0-k*(t-t0)))
            )
            amplitude = scale #(2/PI)**1/4 / (1+2j*(t-t0))**1/2
            if c == "r":
                return (amplitude*wave_part).real
            if c == "i":
                return (amplitude*wave_part).imag
        
        graphre = FunctionGraph(lambda x: get_gaussian_func(x))
        graphre.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="r",t=ttr.get_value(),**kwargs)))
        graphre.add_updater(lambda v: v.shift(shift))
        
        graphim = FunctionGraph(lambda x: get_gaussian_func(x))
        graphim.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="i",t=ttr.get_value(),**kwargs), color=BLUE))
        graphim.add_updater(lambda v: v.shift(shift))
        return VGroup(graphre,graphim)

    def get_wave_sum(self,ttr=None, shift = 0,**kwargs):
        def get_gaussian_func(x,c="r",t=0,k=[-2,-1,1,2],x0=0,t0=0, scale = 1):
            wave_part = np.exp(complex(0, k[0]*(x-x0-k[0]*(t-t0)))) \
                       +np.exp(complex(0, k[1]*(x-x0-k[1]*(t-t0)))) \
                       +np.exp(complex(0, k[2]*(x-x0-k[2]*(t-t0)))) \
                       +np.exp(complex(0, k[3]*(x-x0-k[3]*(t-t0))))
            amplitude = scale #(2/PI)**1/4 / (1+2j*(t-t0))**1/2
            if c == "r":
                return (amplitude*wave_part).real
            if c == "i":
                return (amplitude*wave_part).imag
        
        graphre = FunctionGraph(lambda x: get_gaussian_func(x))
        graphre.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="r",t=ttr.get_value(),**kwargs)))
        graphre.add_updater(lambda v: v.shift(shift))
        
        graphim = FunctionGraph(lambda x: get_gaussian_func(x))
        graphim.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="i",t=ttr.get_value(),**kwargs), color=BLUE))
        graphim.add_updater(lambda v: v.shift(shift))
        return VGroup(graphre,graphim)

#actual animation here of how fourier series adds to a gaussian (mention how resolved and unresolved variables factor in)
class UAWavePacketFourier(Scene): #*
    def construct(self):
        sum1 = TexMobject(
            r"\Psi(x,t)=\sum_{k\in\mathbb{Z}}",
            "c_k",
            r"e^{i\frac{k}{a}(x-\frac{\hbar k}{2ma}t)}"
            )
        sum1.shift(3*UP)
        ck1 = TexMobject(
            r"c_k=",
            r"\frac{\sqrt{\pi}}{2\pi a}e^{-\frac{(k_0-\frac{k}{a})^2}{4}}",
            r"\quad"
            )
        ck1.shift(DOWN)

        dots = VGroup(*[
            Dot(np.array([n,np.exp(-2*(2-n)**2/(4))-3,0]))
            for n in np.linspace(-FRAME_WIDTH/2,FRAME_WIDTH/2,20)
            ])

        wave = self.get_wave( k0 = 10)
        freq1 = self.get_abs(k0 = 2)
        self.add(wave)
        self.play(ShowCreation(sum1))
        self.play(ShowCreation(ck1))
        self.play(ShowCreation(freq1))
        self.play(ShowCreation(dots))

    def get_wave(self,tr=None,**kwargs):
        def get_gaussian_func(x,c="r",t=0,k0=1,x0=0,t0=0, scale = 1):
            wave_part = np.exp(
                complex(0, ((k0+2*(t-t0)*(x-x0)/scale)*(x-x0)/scale-k0**2*(t-t0)/2)/(1+4*(t-t0)**2))
            )
            bell_part = np.exp(-((x-x0)/scale-k0*(t-t0))**2/(1+4*(t-t0)**2))
            amplitude = 1 / (1+2j*(t-t0))**(1/2) #(2/PI)**1/4 / (1+2j*(t-t0))**1/2
            if c == "r":
                return (amplitude*wave_part*bell_part).real+1
            if c == "i":
                return (amplitude*wave_part*bell_part).imag+1
        
        graphre = FunctionGraph(lambda x: get_gaussian_func(x,c="r",t=0,**kwargs))
        
        graphim = FunctionGraph(lambda x: get_gaussian_func(x,c="i",t=0,**kwargs), color=BLUE)
        return VGroup(graphre,graphim)

    def get_abs(self,tr=None,**kwargs):
        def get_gaussian_func(x,t=0,k0=1,x0=0,t0=0, a = 1, scale = 1):
            return np.exp(-2*(a*k0-x)**2/(4*(a)**2))-3
        
        graph = FunctionGraph(lambda x: get_gaussian_func(x,t=0,**kwargs), color = RED)
        
        return graph

class UBWavePacketFourier(Scene): #*
    def construct(self):
        sum1 = TexMobject(
            r"\Psi(x,t)=\sum_{k\in\mathbb{Z}}",
            "c_k",
            r"e^{i\frac{k}{a}(x-\frac{\hbar k}{2ma}t)}"
            )
        sum1.shift(3*UP)
        ck1 = TexMobject(
            r"c_k=",
            r"\frac{\sqrt{\pi}}{2\pi a}e^{-\frac{(k_0-\frac{k}{a})^2}{4}}",
            r"\quad"
            )
        ck1.shift(DOWN)

        sum2 = TexMobject(
            r"\Psi(x,t)=\sum_{k\in\mathbb{Z}}",
            "u_k(t)",
            r"e^{i\frac{k}{a}x}"
            )
        sum2.shift(3*UP)
        ck2 = TexMobject(
            r"u_k(t)=",
            r"\frac{\sqrt{\pi}}{2\pi a}e^{-\frac{(k_0-\frac{k}{a})^2}{4}}",
            r"e^{-i\frac{k}{a}\frac{\hbar k}{2ma}t}"
            )
        ck2.shift(DOWN)

        dots = VGroup(*[
            Dot(np.array([n,np.exp(-2*(2-n)**2/(4))-3,0]))
            for n in np.linspace(-FRAME_WIDTH/2,FRAME_WIDTH/2,20)
            ])

        wave = self.get_wave( k0 = 10)
        freq1 = self.get_abs(k0 = 2)
        freq2 = self.get_uk(k0 = 2)
        self.add(wave,sum1,ck1,freq1,dots)
        self.play(
            Transform(freq1,freq2),
            Transform(ck1[0],ck2[0]),
            Transform(ck1[1],ck2[1]),
            Transform(ck1[2],ck2[2]),
            Transform(sum1[0],sum2[0]),
            Transform(sum1[1],sum2[1]),
            Transform(sum1[2],sum2[2])
            )


    def get_wave(self,tr=None,**kwargs):
        def get_gaussian_func(x,c="r",t=0,k0=1,x0=0,t0=0, scale = 1):
            wave_part = np.exp(
                complex(0, ((k0+2*(t-t0)*(x-x0)/scale)*(x-x0)/scale-k0**2*(t-t0)/2)/(1+4*(t-t0)**2))
            )
            bell_part = np.exp(-((x-x0)/scale-k0*(t-t0))**2/(1+4*(t-t0)**2))
            amplitude = 1 / (1+2j*(t-t0))**(1/2) #(2/PI)**1/4 / (1+2j*(t-t0))**1/2
            if c == "r":
                return (amplitude*wave_part*bell_part).real+1
            if c == "i":
                return (amplitude*wave_part*bell_part).imag+1
        
        graphre = FunctionGraph(lambda x: get_gaussian_func(x,c="r",t=0,**kwargs))
        
        graphim = FunctionGraph(lambda x: get_gaussian_func(x,c="i",t=0,**kwargs), color=BLUE)
        return VGroup(graphre,graphim)

    def get_uk(self,tr=None,**kwargs):
        def get_gaussian_func(x,c="r",t=0,k0=1,x0=0,t0=0, a=1, scale = 1):
            wave_part = np.exp(
                complex(0, x*t)
            )
            bell_part = np.exp(-2*(a*k0-x)**2/(4*(a)**2)) 
            amplitude = 1 / (1+2j*(t-t0))**(1/2) #(2/PI)**1/4 / (1+2j*(t-t0))**1/2
            if c == "r":
                return (amplitude*wave_part*bell_part).real-3
            if c == "i":
                return (amplitude*wave_part*bell_part).imag-3
        
        graphre = FunctionGraph(lambda x: get_gaussian_func(x,c="r",t=0,**kwargs))
        
        graphim = FunctionGraph(lambda x: get_gaussian_func(x,c="i",t=0,**kwargs), color=BLUE)
        return VGroup(graphre,graphim)

    def get_abs(self,tr=None,**kwargs):
        def get_gaussian_func(x,t=0,k0=1,x0=0,t0=0, a = 1, scale = 1):
            return np.exp(-2*(a*k0-x)**2/(4*(a)**2))-3
        
        graph = FunctionGraph(lambda x: get_gaussian_func(x,t=0,**kwargs), color = RED)
        
        return graph

class UCWavePacketFourier(Scene): #*
    def construct(self):
        t=ValueTracker(0)
        sum2 = TexMobject(
            r"\Psi(x,t)=\sum_{k\in\mathbb{Z}}",
            "u_k(t)",
            r"e^{i\frac{k}{a}x}"
            )
        sum2.shift(3*UP)
        ck2 = TexMobject(
            r"u_k(t)=",
            r"\frac{\sqrt{\pi}}{2\pi a}e^{-\frac{(k_0-\frac{k}{a})^2}{4}}",
            r"e^{-i\frac{k}{a}\frac{\hbar k}{2ma}t}"
            )
        ck2.shift(DOWN)

        dots = VGroup(*[
            Dot(np.array([n,np.exp(-2*(2-n)**2/(4))-3,0]))
            for n in np.linspace(-FRAME_WIDTH/2,FRAME_WIDTH/2,20)
            ])

        wave = self.get_wave(tr=t, k0 = 10)
        freq2 = self.get_uk(tr=t,k0 = 2)
        self.add(wave,sum2,ck2,freq2,dots)
        self.play(
            t.set_value,1,
            rate_func=linear,
            run_time=2
            )


    def get_wave(self,tr=None,**kwargs):
        def get_gaussian_func(x,c="r",t=0,k0=1,x0=0,t0=0, scale = 1):
            wave_part = np.exp(
                complex(0, ((k0+2*(t-t0)*(x-x0)/scale)*(x-x0)/scale-k0**2*(t-t0)/2)/(1+4*(t-t0)**2))
            )
            bell_part = np.exp(-((x-x0)/scale-k0*(t-t0))**2/(1+4*(t-t0)**2))
            amplitude = 1 / (1+2j*(t-t0))**(1/2) #(2/PI)**1/4 / (1+2j*(t-t0))**1/2
            if c == "r":
                return (amplitude*wave_part*bell_part).real+1
            if c == "i":
                return (amplitude*wave_part*bell_part).imag+1
        
        graphre = FunctionGraph(lambda x: get_gaussian_func(x,c="r",t=0,**kwargs))
        graphre.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="r",t=tr.get_value(),**kwargs)) )
        
        graphim = FunctionGraph(lambda x: get_gaussian_func(x,c="i",t=0,**kwargs), color=BLUE)
        graphim.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="i",t=tr.get_value(),**kwargs), color=BLUE) )
        return VGroup(graphre,graphim)

    def get_uk(self,tr=None,**kwargs):
        def get_gaussian_func(x,c="r",t=0,k0=1,x0=0,t0=0, a=1, scale = 1):
            wave_part = np.exp(
                complex(0, x*t)
            )
            bell_part = np.exp(-2*(a*k0-x)**2/(4*(a)**2)) 
            amplitude = 1 / (1+2j*(t-t0))**(1/2) #(2/PI)**1/4 / (1+2j*(t-t0))**1/2
            if c == "r":
                return (amplitude*wave_part*bell_part).real-3
            if c == "i":
                return (amplitude*wave_part*bell_part).imag-3
        
        graphre = FunctionGraph(lambda x: get_gaussian_func(x,c="r",t=0,**kwargs))
        graphre.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="r",t=tr.get_value(),**kwargs)) )
        
        graphim = FunctionGraph(lambda x: get_gaussian_func(x,c="i",t=tr.get_value(),**kwargs), color=BLUE)
        graphim.add_updater(lambda v: v.__init__(lambda x: get_gaussian_func(x,c="i",t=tr.get_value(),**kwargs), color=BLUE) )
        return VGroup(graphre,graphim)

class VANLSEReturn(Scene):
    def construct(self):
        NLSE = TexMobject(
            "i\\hbar\\pderiv{\\Psi}{t}+\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi}{x^2}-\\kappa\\left|\\Psi\\right|^2\\Psi=0"
            )

        self.play(Write(NLSE))

class VBNLSEReturn(Scene):
    def construct(self):
        NLSE = TexMobject(
            "i\\hbar\\pderiv{\\Psi}{t}+\\frac{\\hbar^2}{2m}\\pderiv{^2\\Psi}{x^2}-\\kappa\\left|\\Psi\\right|^2\\Psi=0"
            )
        Nonlin = TexMobject(
            "\\text{Nonlinearity from }|\\Psi|^2\\Psi\\text{ term}"
            )
        Nonlin.shift(2*DOWN)
        NoKappa = TexMobject(
            "\\text{Free particle Schrödinger equation if }\\kappa=0"
            )
        NoKappa.shift(3*DOWN)

        self.add(NLSE)
        self.play(Write(Nonlin),Write(NoKappa))

class WBECBox(Scene):
    def construct(self):
        t = ValueTracker(2)

        box = Box(6,6*16/9)

        cloud1 = ProbabilityCloud(color = RED, hbound = 3*16/9, vbound = 3)
        cloud2 = ProbabilityCloud(color = BLUE, hbound = 3*16/9, vbound = 3)
        cloud3 = ProbabilityCloud(color = YELLOW_C, hbound = 3*16/9, vbound = 3)
        cloud4 = ProbabilityCloud(color = GREEN, hbound = 3*16/9, vbound = 3)
        #cloud.move_to(2*LEFT)
        dot_gdw1 = cloud1.gaussian_distribution_wrapper
        dot_gdw1.add_updater(lambda v: v.set_width(t.get_value()))
        dot_gdw1.rotate(TAU/8)
        dot_gdw1.move_to(3*LEFT+2*UP)

        dot_gdw2 = cloud2.gaussian_distribution_wrapper
        dot_gdw2.add_updater(lambda v: v.set_width(t.get_value()))
        dot_gdw2.rotate(TAU/8)
        dot_gdw2.move_to(3*LEFT+2*DOWN)

        dot_gdw3 = cloud3.gaussian_distribution_wrapper
        dot_gdw3.add_updater(lambda v: v.set_width(t.get_value()))
        dot_gdw3.rotate(TAU/8)
        dot_gdw3.move_to(3*RIGHT+2*DOWN)

        dot_gdw4 = cloud4.gaussian_distribution_wrapper
        dot_gdw4.add_updater(lambda v: v.set_width(t.get_value()))
        dot_gdw4.rotate(TAU/8)
        dot_gdw4.move_to(3*RIGHT+2*UP)

        self.add(box)
        self.add(cloud1)
        self.add(cloud2)
        self.add(cloud3)
        self.add(cloud4)
        self.add(dot_gdw1)
        self.add(dot_gdw2)
        self.add(dot_gdw3)
        self.add(dot_gdw4)
        self.play(
            dot_gdw1.shift, RIGHT,
            dot_gdw2.shift, LEFT+UP,
            dot_gdw3.shift, LEFT+UP,
            dot_gdw4.shift, DOWN,
            rate_func=linear,
            run_time=2
            )
        self.play(
            dot_gdw1.shift, RIGHT,
            dot_gdw2.shift, RIGHT+UP,
            dot_gdw3.shift, LEFT+UP,
            dot_gdw4.shift, DOWN,
            rate_func=linear,
            run_time=2
            )
        self.play(
            dot_gdw1.move_to, ORIGIN,
            dot_gdw2.move_to, ORIGIN,
            dot_gdw3.move_to, ORIGIN,
            dot_gdw4.move_to, ORIGIN,
            t.set_value,1,
            rate_func=linear,
            run_time=2
            )
        self.play(
            t.set_value,.25,
            rate_func = linear,
            run_time = 2.5
            )
        self.play(
            dot_gdw1.set_width, .05,
            dot_gdw2.set_width, .05,
            dot_gdw3.set_width, .05,
            dot_gdw4.set_width, .05,
            rate_func=linear,
            run_time=2.5
            )
#BEC image here


class XASpectralNLSE(Scene):
    def construct(self):
        Solve = HeaderMobject("How To Solve It / My Research")
        Solve.shift(UP)
        NLSE = TexMobject(
            "i\\pderiv{\\Psi}{t}", #0
            "+\\pderiv{^2\\Psi}{x^2}", #1
            "-\\kappa|\\Psi|^2\\Psi",  #2
            "=", #3
            "0" #4
            )
        NLSE.shift(2*UP)
        FourierSum = TexMobject(
            "\\Psi(x,t)=\\sum_{k\\in F \\cup G}u_k(t)e^{ikx}"
            )
        #FourierSum.shift(DOWN)

        NLSE1 = TexMobject(
            "\\deriv{u_k}{t}", #0
            "=", #1
            "-ik^2u_k", #2
            "-i\\kappa\\sum_{\\substack{k_1-k_2+k_3=k \\\\ k_1,k_2,k_3,k\\in F \\cup G}}u_{k_1}u^*_{k_2}u_{k_3}" #3
            )


        NLSE1.shift(2*DOWN)

        self.add(Solve)
        self.play(Write(NLSE))

class XBSpectralNLSE(Scene):
    def construct(self):
        Solve = HeaderMobject("How To Solve It / My Research")
        Solve.shift(UP)
        NLSE = TexMobject(
            "i\\pderiv{\\Psi}{t}", #0
            "+\\pderiv{^2\\Psi}{x^2}", #1
            "-\\kappa|\\Psi|^2\\Psi",  #2
            "=", #3
            "0" #4
            )
        NLSE.shift(2*UP)
        FourierSum = TexMobject(
            "\\Psi(x,t)=\\sum_{k\\in F \\cup G}u_k(t)e^{ikx}"
            )
        #FourierSum.shift(DOWN)

        NLSE1 = TexMobject(
            "\\deriv{u_k}{t}", #0
            "=", #1
            "-ik^2u_k", #2
            "-i\\kappa\\sum_{\\substack{k_1-k_2+k_3=k \\\\ k_1,k_2,k_3,k\\in F \\cup G}}u_{k_1}u^*_{k_2}u_{k_3}" #3
            )


        NLSE1.shift(2*DOWN)

        self.add(Solve, NLSE)
        self.play(Write(FourierSum))

class XCSpectralNLSE(Scene):
    def construct(self):
        Solve = HeaderMobject("How To Solve It / My Research")
        Solve.shift(UP)
        NLSE = TexMobject(
            "i\\pderiv{\\Psi}{t}", #0
            "+\\pderiv{^2\\Psi}{x^2}", #1
            "-\\kappa|\\Psi|^2\\Psi",  #2
            "=", #3
            "0" #4
            )
        NLSE.shift(2*UP)
        FourierSum = TexMobject(
            "\\Psi(x,t)=\\sum_{k\\in F \\cup G}u_k(t)e^{ikx}"
            )
        #FourierSum.shift(DOWN)

        NLSE1 = TexMobject(
            "\\deriv{u_k}{t}", #0
            "=", #1
            "-ik^2u_k", #2
            "-i\\kappa\\sum_{\\substack{k_1-k_2+k_3=k \\\\ k_1,k_2,k_3,k\\in F \\cup G}}u_{k_1}u^*_{k_2}u_{k_3}" #3
            )


        NLSE1.shift(2*DOWN)

        self.add(Solve, NLSE, FourierSum)
        self.play(
            ReplacementTransform(NLSE[0].copy(),NLSE1[0]),
            ReplacementTransform(NLSE[1].copy(),NLSE1[2]),
            ReplacementTransform(NLSE[2].copy(),NLSE1[3]),
            ReplacementTransform(NLSE[3].copy(),NLSE1[1]),
            )

class YAMoriZwanzig(Scene): #*
    def construct(self):
        P = TexMobject("P")
        L = TexMobject("\\L")
        Q = TexMobject("Q")
        VGroup(P,L,Q).shift(2*UP)
        P.shift(LEFT)
        L.shift(RIGHT)
        MZ = TexMobject(
            r"\deriv{\uk}{t}=", #0
            r"e^{t\L}P\L\uk^0", #1
            " +", #2
            r"e^{tQ\L}Q\L\uk^0", #3
            "+", #4
            r"\int_0^te^{(t-s)\L}P\L e^{sQ\L}Q\L\uk^0\diff s" #5
            )
        MZTerms = TexMobject(
            "\\underbrace{\\qquad\\qquad}_{markov}", #0
            "\\underbrace{\\qquad\\qquad}_{noise}", #1
            "\\underbrace{\\qquad\\qquad\\qquad}_{memory}" #2
            )
        MZTerms[0].next_to(MZ[1],DOWN)
        MZTerms[1].next_to(MZ[3],DOWN)
        MZTerms[2].next_to(MZ[5],DOWN)
        
        CMA = TexMobject(
            r"\deriv{Pu_k}{t}=", #0
            r"R^0_k(\uh)", #1
            "+",#2
            r"\sum_{i=1}^n\alpha_i(t)t^iR^i_k(\uh)" #3
            )
        CMA.shift(3*DOWN)

        self.play(Write(P))
        self.play(Write(Q))
        self.play(Write(L))

class YBMoriZwanzig(Scene): #*
    def construct(self):
        P = TexMobject("P")
        L = TexMobject("\\L")
        Q = TexMobject("Q")
        VGroup(P,L,Q).shift(2*UP)
        P.shift(LEFT)
        L.shift(RIGHT)
        MZ = TexMobject(
            r"\deriv{\uk}{t}=", #0
            r"e^{t\L}P\L\uk^0", #1
            " +", #2
            r"e^{tQ\L}Q\L\uk^0", #3
            "+", #4
            r"\int_0^te^{(t-s)\L}P\L e^{sQ\L}Q\L\uk^0\diff s" #5
            )
        MZTerms = TexMobject(
            "\\underbrace{\\qquad\\qquad}_{markov}", #0
            "\\underbrace{\\qquad\\qquad}_{noise}", #1
            "\\underbrace{\\qquad\\qquad\\qquad}_{memory}" #2
            )
        MZTerms[0].next_to(MZ[1],DOWN)
        MZTerms[1].next_to(MZ[3],DOWN)
        MZTerms[2].next_to(MZ[5],DOWN)
        
        CMA = TexMobject(
            r"\deriv{Pu_k}{t}=", #0
            r"R^0_k(\uh)", #1
            "+",#2
            r"\sum_{i=1}^n\alpha_i(t)t^iR^i_k(\uh)" #3
            )
        CMA.shift(3*DOWN)

        self.add(P,Q,L)
        self.play(Write(MZ))
        self.play(Write(MZTerms))

class YCMoriZwanzig(Scene): #*
    def construct(self):
        P = TexMobject("P")
        L = TexMobject("\\L")
        Q = TexMobject("Q")
        VGroup(P,L,Q).shift(2*UP)
        P.shift(LEFT)
        L.shift(RIGHT)
        MZ = TexMobject(
            r"\deriv{\uk}{t}=", #0
            r"e^{t\L}P\L\uk^0", #1
            " +", #2
            r"e^{tQ\L}Q\L\uk^0", #3
            "+", #4
            r"\int_0^te^{(t-s)\L}P\L e^{sQ\L}Q\L\uk^0\diff s" #5
            )
        MZTerms = TexMobject(
            "\\underbrace{\\qquad\\qquad}_{markov}", #0
            "\\underbrace{\\qquad\\qquad}_{noise}", #1
            "\\underbrace{\\qquad\\qquad\\qquad}_{memory}" #2
            )
        MZTerms[0].next_to(MZ[1],DOWN)
        MZTerms[1].next_to(MZ[3],DOWN)
        MZTerms[2].next_to(MZ[5],DOWN)
        
        CMA = TexMobject(
            r"\deriv{Pu_k}{t}=", #0
            r"R^0_k(\uh)", #1
            "+",#2
            r"\sum_{i=1}^n\alpha_i(t)t^iR^i_k(\uh)" #3
            )
        CMA.shift(3*DOWN)

        self.add(P,Q,L, MZ, MZTerms)
        self.play(
            ReplacementTransform(MZ[0].copy(),CMA[0]),
            ReplacementTransform(MZ[1].copy(),CMA[1]),
            ReplacementTransform(MZ[2].copy(),CMA[2]),
            ReplacementTransform(MZ[5].copy(),CMA[3])
            )

class ZACMATerms(Scene): #*
    def construct(self):
        R0 = TexMobject(r"R^0_k(\uh)=Pe^{t\L}P\L\uk^0")
        R1 = TexMobject(r"R^1_k(\uh)=Pe^{t\L}P\L Q\L\uk^0")
        R2 = TexMobject(r"R^2_k(\uh)=Pe^{t\L}P\L[P\L-Q\L]Q\L\uk^0")
        R3 = TexMobject(r"R^3_k(\uh)=Pe^{t\L}P\L[P\L P\L-2P\L Q\L - 2Q\L P\L+Q\L Q\L]Q\L\uk^0")
        VGroup(R0,R1,R2,R3).arrange_submobjects(DOWN)

        self.play(Write(R0))
        self.play(Write(R1))
        self.play(Write(R2))
        self.play(Write(R3))

class ZBConvSum(Scene):
    def construct(self):
        conv1 = TexMobject(r"C_k(\mathbf{f}(\mathbf{u}),\mathbf{g}(\mathbf{u}),\mathbf{h}(\mathbf{u}))&=i\kappa\sum_{\substack{k_1-k_2+k_3=k \\ k_1,k_2,k_3,k\in F \cup G}}f_{k_1}(\mathbf{u})g^*_{k_2}(\mathbf{u})h_{k_3}(\mathbf{u})")
        conv2 = TexMobject(r"\Chk(\mathbf{f}(\mathbf{u}),\mathbf{g}(\mathbf{u}),\mathbf{h}(\mathbf{u}))&=i\kappa\sum_{\substack{k_1-k_2+k_3=k \\ k\in F \\ k_1,k_2,k_3,k\in F \cup G}}f_{k_1}(\mathbf{u})g^*_{k_2}(\mathbf{u})h_{k_3}(\mathbf{u})")
        conv3 = TexMobject(r"\tilde{C}_k(\mathbf{f}(\mathbf{u}),\mathbf{g}(\mathbf{u}),\mathbf{h}(\mathbf{u}))&=i\kappa\sum_{\substack{k_1-k_2+k_3=k \\ k\in G \\ k_1,k_2,k_3,k\in F \cup G}}f_{k_1}(\mathbf{u})g^*_{k_2}(\mathbf{u})h_{k_3}(\mathbf{u})")
        VGroup(conv1,conv2,conv3).arrange_submobjects(DOWN)

        self.play(Write(conv1))
        self.play(Write(conv2))
        self.play(Write(conv3))

class ZCRewriteNLSE(Scene):
    def construct(self):
        ddt = TexMobject(
            r"\deriv{u_k}{t} =-ik^2u_k-i\kappa C_k(\u,\u,\u)"
            )
        self.play(Write(ddt))

class ZDPQL(Scene): #*
    def construct(self):
        P = TexMobject(r"Ph(\mathbf{u}^0)=Ph(\uh^0,\ut^0)=h(\uh^0,0)")
        Q = TexMobject(r"Q=1-P")
        L = TexMobject(r"\L =\sum_{k\in F \cup G}\bigg(-ik^2u^0_k-i\kappa C_k(\u^0,\u^0,\u^0)\bigg)\pderiv{}{u^0_k}")
        VGroup(P,Q,L).arrange_submobjects(DOWN)

        self.play(Write(P))
        self.play(Write(Q))
        self.play(Write(L))

#Show screenshot of Mathematica code here

class ZER0(Scene):
    def construct(self):
        R0 = TexMobject(r"R^0_k(\uh)=-ik^2\uhk-\Chk(\uh,\uh,\uh)")

        self.play(Write(R0))

class ZFR1(Scene):
    def construct(self):
        R1 = TexMobject(r"R^1_k(\uh)=2\Chk(\Ct(\uh,\uh,\uh),\uh,\uh)+\Chk(\uh,\Ct(\uh,\uh,\uh),\uh)")

        self.play(Write(R1))

class ZGR2(Scene):
    def construct(self):
        R2 = TexMobject(r"R^2_k(\uh)=",
                        r"2\Chk(\uh,\uh,ik^2\Ct(\uh,\uh,\uh)-\Ct(\uh,\uh,-ik^2\uh-\Ch(\uh,\uh,\uh))+2\Ct(\uh,\uh,\Ct(\uh,\uh,\uh))",
                        r"\qquad\quad +\Ct(\uh,ik^2\uh+\Ch(\uh,\uh,\uh),\uh)-\Ct(-ik^2\uh-\Ch(\uh,\uh,\uh),\uh,\uh))",
                        r"+4\Chk(\uh,\uh,2\Ct(\uh,\uh,-ik^2\uh-\Ch(\uh,\uh,\uh))+\Ct(\uh,-ik^2\uh-\Ch(\uh,\uh,\uh),\uh))",
                        r"+\Chk(\uh,ik^2\Ct(\uh,\uh,\uh)-\Ct(\uh,\uh,-ik^2\uh-\Ch(\uh,\uh,\uh))+2\Ct(\uh,\uh,\Ct(\uh,\uh,\uh))",
                        r"\qquad\quad +\Ct(\uh,ik^2\uh+\Ch(\uh,\uh,\uh)+\Ct(\uh,\uh,\uh),\uh)-\Ct(-ik^2\uh-\Ch(\uh,\uh,\uh),\uh,\uh),\uh)",
                        r"+2\Chk(\uh,2\Ct(\uh,\uh,-ik^2-\Ch(\uh,\uh,\uh))+\Ct(\uh,-ik^2-\Ch(\uh,\uh,\uh),\uh),\uh)",
                        r"+2\Chk(\Ct(\uh,\uh,\uh),\uh,\Ct(\uh,\uh,\uh))",
                        r"+4\Chk(\Ct(\uh,\uh,\uh),\Ct(\uh,\uh,\uh),\uh)")
        R2.scale(.7)
        R2[0].to_corner(UL)
        for n in range(8):
            R2[n+1].next_to(R2[0],RIGHT)
            R2[n+1].shift(3*n*DOWN/4)

        
        self.play(Write(R2))

#Show screenshot of R3 Mathematica here

#Show screenshot of MATLAB Plots

class ZHAcknowledgements(Scene):
    def construct(self):
        Thx = HeaderMobject("Acknowledgements")
        Names1 = TextMobject(
            "David Latimer",
            "Jake Price",
            "Grant Sanderson",
            "Ti Nguyen")
        Names2 = TextMobject(
            "Alison Tracy Hale",
            "Aislinn Melchior",
            "Friends and Family"
            )
        #Names[0].to_corner(UL)
        #for n in range(5):
        #    Names[n+1].next_to(Names[0],RIGHT)
        #    Names[n+1].shift(3*n*DOWN/4)
        VGroup(Names1[0],Names1[1],Names1[2],Names1[3],Names2[0],Names2[1],Names2[2]).arrange_submobjects(DOWN).shift(DOWN)

        self.add(Thx)
        self.add(Names1,Names2)
        self.wait(1)
