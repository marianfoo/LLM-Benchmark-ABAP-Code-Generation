CLASS z_humaneval_105 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_digit_map,
             digit TYPE i,
             name  TYPE string,
           END OF ty_digit_map.
    TYPES: tt_digit_map TYPE TABLE OF ty_digit_map,
           tt_integers  TYPE TABLE OF i,
           tt_strings   TYPE TABLE OF string.

    CLASS-METHODS: by_length
      IMPORTING iv_arr           TYPE tt_integers
      RETURNING VALUE(rv_result) TYPE string_table.

ENDCLASS.

CLASS z_humaneval_105 IMPLEMENTATION.

  METHOD by_length.
    DATA: lt_digit_map TYPE tt_digit_map,
          ls_digit_map TYPE ty_digit_map,
          lt_filtered  TYPE tt_integers,
          lv_int       TYPE i,
          lv_string    TYPE string.

    ls_digit_map-digit = 1. ls_digit_map-name = 'One'. APPEND ls_digit_map TO lt_digit_map.
    ls_digit_map-digit = 2. ls_digit_map-name = 'Two'. APPEND ls_digit_map TO lt_digit_map.
    ls_digit_map-digit = 3. ls_digit_map-name = 'Three'. APPEND ls_digit_map TO lt_digit_map.
    ls_digit_map-digit = 4. ls_digit_map-name = 'Four'. APPEND ls_digit_map TO lt_digit_map.
    ls_digit_map-digit = 5. ls_digit_map-name = 'Five'. APPEND ls_digit_map TO lt_digit_map.
    ls_digit_map-digit = 6. ls_digit_map-name = 'Six'. APPEND ls_digit_map TO lt_digit_map.
    ls_digit_map-digit = 7. ls_digit_map-name = 'Seven'. APPEND ls_digit_map TO lt_digit_map.
    ls_digit_map-digit = 8. ls_digit_map-name = 'Eight'. APPEND ls_digit_map TO lt_digit_map.
    ls_digit_map-digit = 9. ls_digit_map-name = 'Nine'. APPEND ls_digit_map TO lt_digit_map.

    LOOP AT iv_arr INTO lv_int.
      IF lv_int >= 1 AND lv_int <= 9.
        APPEND lv_int TO lt_filtered.
      ENDIF.
    ENDLOOP.

    SORT lt_filtered BY table_line DESCENDING.

    LOOP AT lt_filtered INTO lv_int.
      READ TABLE lt_digit_map INTO ls_digit_map WITH KEY digit = lv_int.
      IF sy-subrc = 0.
        APPEND ls_digit_map-name TO rv_result.
      ENDIF.
    ENDLOOP.

  ENDMETHOD.

ENDCLASS.