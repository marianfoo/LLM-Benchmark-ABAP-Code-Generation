CLASS z_humaneval_045 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS triangle_area
      IMPORTING
        a             TYPE f
        h             TYPE f
      RETURNING
        VALUE(result) TYPE f.

ENDCLASS.

CLASS z_humaneval_045 IMPLEMENTATION.

  METHOD triangle_area.
    result = a * h / 2.
  ENDMETHOD.

ENDCLASS.