"""写真ブースの案内に使う、画像ファイル不要の小さな線画。"""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class BoothIllustration(QWidget):
    def __init__(self, motif: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.motif = motif
        self.setFixedSize(64, 64)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.scale(self.width() / 64, self.height() / 64)
        painter.setPen(QPen(QColor("#94765e"), 1.4, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if self.motif == "flowers":
            self._flowers(painter)
        elif self.motif == "camera":
            path = QPainterPath(QPointF(9, 22))
            for x, y in ((20, 22), (23, 16), (36, 16), (40, 22), (54, 22)):
                path.lineTo(x, y)
            path.quadTo(57, 22, 57, 26)
            path.lineTo(57, 49)
            path.quadTo(57, 52, 53, 52)
            path.lineTo(10, 52)
            path.quadTo(6, 52, 6, 48)
            path.lineTo(6, 26)
            path.quadTo(6, 22, 9, 22)
            painter.drawPath(path)
            painter.drawEllipse(QRectF(22, 28, 21, 21))
            painter.drawEllipse(QRectF(26, 32, 13, 13))
            painter.drawRoundedRect(QRectF(11, 17, 7, 5), 1, 1)
            painter.drawLine(47, 27, 51, 27)
            self._rays(painter)
        elif self.motif == "people":
            painter.drawEllipse(QRectF(24, 15, 16, 16))
            painter.drawEllipse(QRectF(7, 19, 14, 14))
            painter.drawEllipse(QRectF(43, 19, 14, 14))
            for left, top, width in ((3, 34, 23), (39, 34, 23), (17, 32, 30)):
                path = QPainterPath(QPointF(left, 55))
                path.cubicTo(left, top - 6, left + width, top - 6, left + width, 55)
                path.closeSubpath()
                painter.setBrush(QColor("#f8f1eb"))
                painter.drawPath(path)
            self._rays(painter)
        elif self.motif == "photo":
            painter.save()
            painter.translate(32, 34)
            painter.rotate(10)
            painter.drawRect(QRectF(-18, -24, 36, 48))
            painter.fillRect(QRectF(-14, -20, 28, 30), QColor("#e7d8cd"))
            painter.drawRect(QRectF(-14, -20, 28, 30))
            painter.restore()
        elif self.motif == "printer":
            painter.drawRoundedRect(QRectF(16, 9, 30, 15), 1, 1)
            painter.drawRoundedRect(QRectF(5, 23, 53, 23), 3, 3)
            painter.drawEllipse(QRectF(49, 28, 3, 3))
            painter.setBrush(QColor("#f8f1eb"))
            path = QPainterPath(QPointF(18, 38))
            for x, y in ((44, 38), (40, 61), (14, 61)):
                path.lineTo(x, y)
            path.closeSubpath()
            painter.drawPath(path)
            painter.fillRect(QRectF(21, 42, 16, 12), QColor("#ddc9ba"))
            painter.drawLine(51, 51, 56, 54)
            painter.drawLine(48, 56, 50, 61)
        painter.end()

    @staticmethod
    def _rays(painter: QPainter) -> None:
        for x1, y1, x2, y2 in ((29, 8, 27, 2), (35, 7, 36, 1), (41, 10, 45, 5)):
            painter.drawLine(x1, y1, x2, y2)

    @staticmethod
    def _flowers(painter: QPainter) -> None:
        for x, y, color, lean in ((21, 17, "#d9b8ae", 6), (42, 27, "#c8bfd3", -5),
                                  (34, 10, "#dcc785", 1)):
            painter.setPen(QPen(QColor("#a3a189"), 0.8))
            path = QPainterPath(QPointF(x, y + 4))
            path.quadTo(x + lean, 43, 33, 62)
            painter.drawPath(path)
            painter.drawLine(x + lean / 2, 42, x + lean / 2 - 7, 36)
            painter.save()
            painter.translate(x, y)
            painter.setPen(Qt.PenStyle.NoPen)
            petal = QColor(color)
            petal.setAlpha(175)
            painter.setBrush(petal)
            for _ in range(7):
                painter.drawEllipse(QRectF(-2.5, -9, 5, 9))
                painter.rotate(360 / 7)
            painter.setBrush(QColor("#b29a6d"))
            painter.drawEllipse(QRectF(-2, -2, 4, 4))
            painter.restore()
