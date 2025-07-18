CLASS z_humaneval_144 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: simplify
      IMPORTING
        VALUE(x)      TYPE string
        VALUE(n)      TYPE string
      RETURNING
        VALUE(result) TYPE abap_bool.

ENDCLASS.

CLASS z_humaneval_144 IMPLEMENTATION.

  METHOD simplify.
    DATA: lv_a               TYPE string,
          lv_b               TYPE string,
          lv_c               TYPE string,
          lv_d               TYPE string,
          lv_numerator       TYPE i,
          lv_denom           TYPE i,
          lv_division_result TYPE f.

    SPLIT x AT '/' INTO lv_a lv_b.
    SPLIT n AT '/' INTO lv_c lv_d.

    lv_numerator = lv_a * lv_c.
    lv_denom = lv_b * lv_d.

    lv_division_result = lv_numerator / lv_denom.

    IF lv_division_result = trunc( lv_division_result ).
      result = abap_true.
    ELSE.
      result = abap_false.
    ENDIF.

  ENDMETHOD.

ENDCLASS.