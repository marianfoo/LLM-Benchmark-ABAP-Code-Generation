CLASS ltcl_customer_id_generator DEFINITION FINAL FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.

    METHODS:
      setup,
      teardown,

      " Test main method
      test_generate_success FOR TESTING,
      test_generate_empty FOR TESTING,
      test_generate_unknown FOR TESTING,
      test_generate_spaces FOR TESTING,
      test_generate_large_id FOR TESTING,
      test_generate_zero_id FOR TESTING,

      " Helper methods for test data setup
      setup_test_data,
      cleanup_test_data.

ENDCLASS.

CLASS ltcl_customer_id_generator IMPLEMENTATION.

  METHOD setup.
    " Setup test environment
    setup_test_data( ).
  ENDMETHOD.

  METHOD teardown.
    " Cleanup test environment
    cleanup_test_data( ).
  ENDMETHOD.

  METHOD setup_test_data.
    " Declare internal tables
    DATA lt_country_codes TYPE STANDARD TABLE OF zcountry_codes.
    DATA lt_region_codes  TYPE STANDARD TABLE OF zregion_codes.
    DATA lt_customer_types TYPE STANDARD TABLE OF zcustomer_types.

    " Fill country data
    lt_country_codes = VALUE #(
      ( country_name = 'GERMANY' country_code = '0049' iso_code = 'DE' phone_prefix = '+49' )
      ( country_name = 'AUSTRIA' country_code = '0043' iso_code = 'AT' phone_prefix = '+43' )
      ( country_name = 'SWITZERLAND' country_code = '0041' iso_code = 'CH' phone_prefix = '+41' )
      ( country_name = 'FRANCE' country_code = '0033' iso_code = 'FR' phone_prefix = '+33' )
    ).
    INSERT zcountry_codes FROM TABLE lt_country_codes.

    " Fill region data
    lt_region_codes = VALUE #(
      ( country_code = '0049' region_name = 'NORTH RHINE-WESTPHALIA' region_code = 'NW' region_type = 'STATE' )
      ( country_code = '0049' region_name = 'BAVARIA' region_code = 'BY' region_type = 'STATE' )
      ( country_code = '0043' region_name = 'VIENNA' region_code = 'VI' region_type = 'STATE' )
      ( country_code = '0041' region_name = 'ZURICH' region_code = 'ZH' region_type = 'CANTON' )
    ).
    INSERT zregion_codes FROM TABLE lt_region_codes.

    " Fill customer type data
    lt_customer_types = VALUE #(
      ( customer_type = 'BUSINESS_CUSTOMER' type_code = 'B' description = 'Business Customer' )
      ( customer_type = 'PRIVATE_CUSTOMER' type_code = 'P' description = 'Private Customer' )
      ( customer_type = 'GOVERNMENT_CUSTOMER' type_code = 'G' description = 'Government Customer' )
    ).
    INSERT zcustomer_types FROM TABLE lt_customer_types.

    COMMIT WORK.
  ENDMETHOD.


  METHOD cleanup_test_data.
    " Clean up test data
    DELETE FROM zcountry_codes.
    DELETE FROM zregion_codes.
    DELETE FROM zcustomer_types.
    COMMIT WORK.
  ENDMETHOD.

  METHOD test_generate_success.
    " Test successful customer ID generation
    DATA(lv_result) = z_humaneval_erp_008=>generate_customer_id(
      country   = 'Germany'
      region = 'North Rhine-Westphalia'
      customer_type   = 'Private_Customer'
      digit_id   = 123
    ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = '0049NWP00123'
      msg = 'Customer ID should be generated correctly'
    ).

    cl_abap_unit_assert=>assert_equals(
      act = strlen( lv_result )
      exp = 12
      msg = 'Customer ID should be exactly 12 characters'
    ).
  ENDMETHOD.

  METHOD test_generate_empty.
    " Test with empty input values
    DATA(lv_result) = z_humaneval_erp_008=>generate_customer_id(
      country   = ''
      region = ''
      customer_type   = ''
      digit_id   = 456
    ).

    cl_abap_unit_assert=>assert_equals(
      act = strlen( lv_result )
      exp = 12
      msg = 'Customer ID should still be 12 characters with empty input'
    ).

    " Should contain initial values and formatted ID
    cl_abap_unit_assert=>assert_char_cp(
      act = lv_result
      exp = '*00456'
      msg = 'Should contain formatted ID at the end'
    ).
  ENDMETHOD.

  METHOD test_generate_unknown.
    " Test with unknown country, region, and customer type
    DATA(lv_result) = z_humaneval_erp_008=>generate_customer_id(
      country   = 'Unknown Country'
      region = 'Unknown Region'
      customer_type   = 'Unknown Type'
      digit_id   = 789
    ).

    cl_abap_unit_assert=>assert_equals(
      act = strlen( lv_result )
      exp = 12
      msg = 'Customer ID should be 12 characters even with unknown values'
    ).

    cl_abap_unit_assert=>assert_char_cp(
      act = lv_result
      exp = '*00789'
      msg = 'Should contain formatted ID at the end'
    ).
  ENDMETHOD.

  METHOD test_generate_spaces.
    " Test with input containing spaces and special characters
    DATA(lv_result) = z_humaneval_erp_008=>generate_customer_id(
      country   = '  Germany  '
      region = '  North Rhine-Westphalia  '
      customer_type   = '  Private_Customer  '
      digit_id   = 999
    ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = '0049NWP00999'
      msg = 'Should handle spaces correctly'
    ).
  ENDMETHOD.

  METHOD test_generate_large_id.
    " Test with large ID number
    DATA(lv_result) = z_humaneval_erp_008=>generate_customer_id(
      country   = 'Germany'
      region = 'Bavaria'
      customer_type   = 'Business_Customer'
      digit_id   = 99999
    ).

    cl_abap_unit_assert=>assert_char_cp(
      act = lv_result
      exp = '*99999'
      msg = 'Should handle large ID numbers correctly'
    ).
  ENDMETHOD.

  METHOD test_generate_zero_id.
    " Test with zero ID
    DATA(lv_result) = z_humaneval_erp_008=>generate_customer_id(
      country   = 'Austria'
      region = 'Vienna'
      customer_type   = 'Government_Customer'
      digit_id   = 0
    ).

    cl_abap_unit_assert=>assert_char_cp(
      act = lv_result
      exp = '*00000'
      msg = 'Should format zero ID with leading zeros'
    ).
  ENDMETHOD.


ENDCLASS.