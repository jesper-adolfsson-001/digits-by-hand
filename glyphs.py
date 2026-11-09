"""Parametric pen movements in a unit box, independent of any digit bitmap."""
import numpy as np


def curve(start, *segments):
    """Concatenate cubic Bezier pen movements; each segment ends at its last point."""
    t = np.linspace(0, 1, 7)[:, None]
    path = [start]
    for p, q, end in segments:
        a, b, c, d = np.asarray([path[-1], p, q, end])
        points = (1-t)**3*a + 3*(1-t)**2*t*b + 3*(1-t)*t*t*c + t**3*d
        path.extend(points[1:])
    return np.asarray(path)


def movements(digit, rng):
    u = rng.uniform
    m = u(.33,.62)  # Height at which the pen changes lobe or direction.
    if digit == 0:
        t = np.linspace(0, 2*np.pi, 28)
        return [np.column_stack((.5+.5*np.cos(t), .5+.5*np.sin(t)))]
    if digit == 1:
        flag = u(.0,.85)
        stem = u(.35,.75)
        paths = [curve((stem-flag,u(.22,.69)), ((stem-.3*flag,.30),(stem,.10),(stem,0)),
                       ((stem,m),(stem,1),(stem+u(-.23,.23),1)))]
        if rng.random() < .22:
            paths.append(np.array([[.0,1],[1,1]]))
        return paths
    if digit == 2:
        return [curve((0,u(.03,.36)), ((0,-.04),(.60,-.08),(.73,.05)),
                      ((1,.20),(.77,.40),(.46,m)),
                      ((.30,.69),(0,.91),(.03,.97)),
                      ((.23,u(.75,1.1)),(.78,.94),(1,u(.88,1.12))))]
    if digit == 3:
        if rng.random() < .55:
            # A cursive 3 can turn back across its own upper stroke like
            # the top of an ezh, with a deep open hook at the bottom.
            lobe = u(.22,.61)
            return [curve((u(-.1,.2),u(.00,.19)),
                          ((.13,-.08),(lobe+.15,-.01),(lobe,.18)),
                          ((lobe-.04,.27),(-.08,m+.02),(0,m)),
                          ((.14,m-.16),(.50,m-.13),(.76,m+.04)),
                          ((1.15,.70),(1.02,.96),(.75,.99)),
                          ((.45,1.06),(.11,.96),(.06,u(.60,.91))))]
        midx=u(.08,.60)
        tip=u(.58,1.03)
        return [curve((0,u(.00,.16)), ((.36,-.10),(.88,-.06),(.85,.14)),
                      ((.85,m-.16),(midx+.15,m-.03),(midx,m)),
                      ((midx+.22,m-.06),(1.04,m+.02),(1,.78)),
                      ((1.02,1.07),(.16,1.12),(0,tip)))]
    if digit == 4:
        stem=u(.52,.90)
        cross=u(.48,.69)
        return [curve((u(.05,.83),0), ((.15,.24),(-.04,cross),(.04,cross)),
                      ((.20,cross+.06),(.72,cross),(1,cross-.03))),
                np.array([[stem,u(0,.34)],[stem+u(-.32,.08),1]])]
    if digit == 5:
        return [curve((1,0), ((.73,.02),(.32,0),(.15,.04)),
                      ((.12,.18),(.03,.40),(0,m)),
                      ((.42,m-.17),(.98,m-.10),(.95,.72)),
                      ((.93,.94),(.38,1.04),(0,u(.87,1.04))))]
    if digit == 6:
        return [curve((u(.53,1),0), ((.32,.01),(.03,.32),(0,.65)),
                      ((-.06,.94),(.27,1.02),(.54,1)),
                      ((.98,.99),(1.1,.71),(.88,m+.04)),
                      ((.63,m-.09),(.21,m+.02),(.06,.70)))]
    if digit == 7:
        tip=u(0,.48)
        paths = [curve((0,tip), ((0,.08),(.18,-.02),(.69,0)),
                       ((.95,-.04),(1,.02),(.90,.25)),
                       ((.77,.52),(.47,.76),(.37,1)))]
        if rng.random() < .6:
            paths.append(np.array([[u(-.1,.3),m+.06],[u(.8,1.05),m-.05]]))
        return paths
    if digit == 8:
        # An 8 can have very different-sized loops whose centers don't line up.
        xc=u(.25,.70)
        upper=u(.28,.92)
        lower=u(.03,.76)
        return [curve((xc,m), ((upper-.55,m-.09),(upper-.68,0),(upper,0)),
                      ((upper+.60,0),(upper+.53,m-.16),(xc,m)),
                      ((lower-.56,m+.15),(lower-.58,1),(lower,1)),
                      ((lower+.58,1),(lower+.59,m+.17),(xc,m)))]
    # A 9 may descend straight, or swing outward and curl back along the base.
    cx=u(.28,.55)
    right=cx+.37
    tail=u(-.18,.88)
    return [curve((right,.19), ((right-.04,-.10),(cx-.40,-.05),(cx-.42,.19)),
                  ((cx-.60,m+.09),(right,m+.12),(right,.14)),
                  ((right,m),(right+tail,.83),(right+tail,.91)),
                  ((right+tail,1),(right+tail-.13,1),(right+tail-u(.02,.76),1)))]
