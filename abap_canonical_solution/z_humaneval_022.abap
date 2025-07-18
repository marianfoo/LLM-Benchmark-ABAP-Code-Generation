CLASS z_humaneval_022 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_any_value,
             value TYPE string,
           END OF ty_any_value.
    TYPES: tt_any_values TYPE TABLE OF ty_any_value.

    CLASS-METHODS: filter_integers
      IMPORTING
        values        TYPE tt_any_values
      RETURNING
        VALUE(result) TYPE int4_table.

  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_022 IMPLEMENTATION.
  METHOD filter_integers.
    DATA: lv_int TYPE i.

    LOOP AT values INTO DATA(wa_value).
      TRY.
          lv_int = wa_value-value.
          IF wa_value-value CO '0123456789-+'.
            result = VALUE #( BASE result ( lv_int ) ).
          ENDIF.
        CATCH cx_sy_conversion_no_number.
          CONTINUE.
      ENDTRY.
    ENDLOOP.
  ENDMETHOD.
ENDCLASS.