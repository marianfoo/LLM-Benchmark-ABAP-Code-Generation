CLASS z_humaneval_092 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: any_int
      IMPORTING
        x             TYPE decfloat34
        y             TYPE decfloat34
        z             TYPE decfloat34
      RETURNING
        VALUE(result) TYPE abap_bool.
  PROTECTED SECTION.
  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_092 IMPLEMENTATION.

  METHOD any_int.
    DATA: x_int TYPE i,
          y_int TYPE i,
          z_int TYPE i,
          x_str TYPE string,
          y_str TYPE string,
          z_str TYPE string.

    x_str = |{ x }|.
    y_str = |{ y }|.
    z_str = |{ z }|.

    IF x_str CS '.' OR y_str CS '.' OR z_str CS '.'.
      result = abap_false.
      RETURN.
    ENDIF.

    IF x = trunc( x ) AND y = trunc( y ) AND z = trunc( z ).
      x_int = x.
      y_int = y.
      z_int = z.

      IF ( x_int + y_int = z_int ) OR
         ( x_int + z_int = y_int ) OR
         ( y_int + z_int = x_int ).
        result = abap_true.
      ELSE.
        result = abap_false.
      ENDIF.
    ELSE.
      result = abap_false.
    ENDIF.
  ENDMETHOD.

ENDCLASS.