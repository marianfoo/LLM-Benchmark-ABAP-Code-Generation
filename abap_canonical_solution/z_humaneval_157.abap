CLASS z_humaneval_157 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: right_angle_triangle
      IMPORTING
        a             TYPE i
        b             TYPE i
        c             TYPE i
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_157 IMPLEMENTATION.

  METHOD right_angle_triangle.
    IF a * a = b * b + c * c OR
       b * b = a * a + c * c OR
       c * c = a * a + b * b.
      result = abap_true.
    ELSE.
      result = abap_false.
    ENDIF.
  ENDMETHOD.

ENDCLASS.