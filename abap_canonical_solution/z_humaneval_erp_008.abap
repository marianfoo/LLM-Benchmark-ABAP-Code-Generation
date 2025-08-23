CLASS z_humaneval_erp_008 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: generate_customer_id
      IMPORTING
        country               TYPE char50
        region                TYPE char100
        customer_type         TYPE char30
        digit_id              TYPE i
      RETURNING
        VALUE(rv_customer_id) TYPE string.

  PRIVATE SECTION.
    CLASS-METHODS:
      get_country_code
        IMPORTING
          iv_country_name        TYPE char50
        RETURNING
          VALUE(rv_country_code) TYPE char4,

      get_region_code
        IMPORTING
          iv_country_code       TYPE char4
          iv_region_name        TYPE char100
        RETURNING
          VALUE(rv_region_code) TYPE char2,

      get_customer_type_code
        IMPORTING
          iv_customer_type    TYPE char30
        RETURNING
          VALUE(rv_type_code) TYPE char1,

      extract_region_abbreviation
        IMPORTING
          iv_region_name         TYPE char100
        RETURNING
          VALUE(rv_abbreviation) TYPE char2.

ENDCLASS.

CLASS z_humaneval_erp_008 IMPLEMENTATION.

  METHOD generate_customer_id.

    DATA: lv_country_code          TYPE char4,
          lv_region_code           TYPE char2,
          lv_type_code             TYPE char1,
          lv_id_formatted          TYPE string,
          lv_country_trimmed       TYPE char50,
          lv_region_trimmed        TYPE char100,
          lv_customer_type_trimmed TYPE char30.

    " Trim input parameters
    lv_country_trimmed = country.
    lv_region_trimmed = region.
    lv_customer_type_trimmed = customer_type.

    CONDENSE: lv_country_trimmed,
              lv_region_trimmed,
              lv_customer_type_trimmed.

    " Get country code
    lv_country_code = get_country_code( lv_country_trimmed ).

    " Get region code
    lv_region_code = get_region_code(
      iv_country_code = lv_country_code
      iv_region_name = lv_region_trimmed ).

    " Get customer type code
    lv_type_code = get_customer_type_code( lv_customer_type_trimmed ).

    " Format ID with leading zeros (5 digits)
    lv_id_formatted = |{ digit_id WIDTH = 5 PAD = '0' ALIGN = RIGHT }|.

    " Combine all parts to create 12-character customer ID
    rv_customer_id = |{ lv_country_code }{ lv_region_code }{ lv_type_code }{ lv_id_formatted }|.

  ENDMETHOD.

  METHOD get_country_code.

    DATA: lv_country_upper TYPE char50.

    " Convert to uppercase for comparison
    lv_country_upper = iv_country_name.
    TRANSLATE lv_country_upper TO UPPER CASE.
    CONDENSE lv_country_upper.

    " Try to get country code from table
    SELECT SINGLE country_code
      FROM zcountry_codes
      WHERE upper( country_name ) = @lv_country_upper
      INTO @rv_country_code.

    " If not found, return initial value
    IF sy-subrc <> 0.
      rv_country_code = 'XXXX'.
    ENDIF.

  ENDMETHOD.

  METHOD get_region_code.

    DATA: lv_region_upper TYPE char100.

    " Convert to uppercase for comparison
    lv_region_upper = iv_region_name.
    TRANSLATE lv_region_upper TO UPPER CASE.
    CONDENSE lv_region_upper.

    " Try to get region code from table
    SELECT SINGLE region_code
      FROM zregion_codes
      WHERE country_code = @iv_country_code
        AND upper( region_name ) = @lv_region_upper
      INTO @rv_region_code.

    " If not found, return initial value
    IF sy-subrc <> 0.
      rv_region_code = 'XX'.
    ENDIF.

  ENDMETHOD.

  METHOD get_customer_type_code.

    DATA: lv_type_upper TYPE char30.

    " Convert to uppercase for comparison
    lv_type_upper = iv_customer_type.
    TRANSLATE lv_type_upper TO UPPER CASE.
    CONDENSE lv_type_upper.

    " Try to get type code from table
    SELECT SINGLE type_code
      FROM zcustomer_types
      WHERE upper( customer_type ) = @lv_type_upper
      INTO @rv_type_code.

    " If not found, return initial value
    IF sy-subrc <> 0.
      rv_type_code = 'X'.
    ENDIF.

  ENDMETHOD.

  METHOD extract_region_abbreviation.

    DATA: lt_words       TYPE TABLE OF string,
          lv_word        TYPE string,
          lv_temp_abbrev TYPE string.

    " Handle empty or invalid input
    IF iv_region_name IS INITIAL.
      rv_abbreviation = 'XX'.
      RETURN.
    ENDIF.

    " Split region name into words
    SPLIT iv_region_name AT space INTO TABLE lt_words.

    " Extract first letter of each word (max 2 characters)
    LOOP AT lt_words INTO lv_word.
      IF strlen( lv_word ) > 0.
        lv_temp_abbrev = |{ lv_temp_abbrev }{ lv_word+0(1) }|.
        " Stop after 2 characters
        IF strlen( lv_temp_abbrev ) >= 2.
          EXIT.
        ENDIF.
      ENDIF.
    ENDLOOP.

    " Ensure we have exactly 2 characters
    IF strlen( lv_temp_abbrev ) = 0.
      rv_abbreviation = 'XX'.
    ELSEIF strlen( lv_temp_abbrev ) = 1.
      rv_abbreviation = |{ lv_temp_abbrev }X|.
    ELSE.
      rv_abbreviation = lv_temp_abbrev+0(2).
    ENDIF.

    " Convert to uppercase
    TRANSLATE rv_abbreviation TO UPPER CASE.

  ENDMETHOD.

ENDCLASS.